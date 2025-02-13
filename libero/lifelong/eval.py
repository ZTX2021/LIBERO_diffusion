import argparse
import sys
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import hydra
import json
import numpy as np
import pprint
import time
import torch
import wandb
import yaml
import multiprocessing
from easydict import EasyDict
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoModel, pipeline, AutoTokenizer, logging
from pathlib import Path

from libero.libero import get_libero_path, benchmark
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
from libero.libero.utils.time_utils import Timer
from libero.libero.utils.video_utils import VideoWriter
from libero.lifelong.algos import *
from libero.lifelong.datasets import get_dataset, SequenceVLDataset, GroupedTaskDataset
from libero.lifelong.metric import (
    evaluate_loss,
    evaluate_success,
    raw_obs_to_tensor_obs,
)
from libero.lifelong.utils import (
    control_seed,
    safe_device,
    torch_load_model,
    NpEncoder,
    compute_flops,
)

from libero.lifelong.main import get_task_embs

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils

import time

from hydra import compose, initialize


benchmark_map = {
    "libero_10": "LIBERO_10",
    "libero_spatial": "LIBERO_SPATIAL",
    "libero_object": "LIBERO_OBJECT",
    "libero_goal": "LIBERO_GOAL",
}

algo_map = {
    "base": "Sequential",
    "er": "ER",
    "ewc": "EWC",
    "packnet": "PackNet",
    "multitask": "Multitask",
}

policy_map = {
    "bc_rnn_policy": "BCRNNPolicy",
    "bc_transformer_policy": "BCTransformerPolicy",
    "bc_vilt_policy": "BCViLTPolicy",
}

initialize(config_path="../../libero/configs")
hydra_cfg = compose(config_name="config")
yaml_config = OmegaConf.to_yaml(hydra_cfg)
cfg = EasyDict(yaml.safe_load(yaml_config))
N_EP = cfg.train.n_epochs // cfg.eval.eval_every + 1
EVAL_EVERY = cfg.eval.eval_every


def eval_one_task(
    run_folder, 
    benchmark_name, 
    policy, 
    alg, 
    seed, 
    load_task, 
    ep, 
    task_id,
    save_dir,
    device_id = 0, 
    save_videos = False
    ):
    device_id = "cuda:" + str(device_id)

    if multiprocessing.get_start_method(allow_none=True) != "spawn":  
        multiprocessing.set_start_method("spawn", force=True)
    
    
    
    try:
        if alg== "multitask":
            model_path = os.path.join(run_folder, f"multitask_model_ep{ep}.pth")
            sd, cfg, previous_mask = torch_load_model(
                model_path, map_location=device_id
            )
        else:
            model_path = os.path.join(run_folder, f"task{load_task}_model_epoch{ep}.pth")
            sd, cfg, previous_mask = torch_load_model(
                model_path, map_location=device_id
            )
    except:
        print(f"[error] cannot find the checkpoint at {str(model_path)}")
        sys.exit(0)

    cfg.folder = get_libero_path("datasets")
    cfg.bddl_folder = get_libero_path("bddl_files")
    cfg.init_states_folder = get_libero_path("init_states")

    cfg.device = device_id
    algo = safe_device(eval(algo_map[alg])(10, cfg), cfg.device)
    algo.policy.previous_mask = previous_mask

    if cfg.lifelong.algo == "PackNet":
        algo.eval()
        for module_idx, module in enumerate(algo.policy.modules()):
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                weight = module.weight.data
                mask = algo.previous_masks[module_idx].to(cfg.device)
                weight[mask.eq(0)] = 0.0
                weight[mask.gt(task_id + 1)] = 0.0
                # we never train norm layers
            if "BatchNorm" in str(type(module)) or "LayerNorm" in str(type(module)):
                module.eval()

    algo.policy.load_state_dict(sd)

    if not hasattr(cfg.data, "task_order_index"):
        cfg.data.task_order_index = 0

    # get the benchmark the task belongs to
    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    descriptions = [benchmark.get_task(i).language for i in range(10)]
    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)

    task = benchmark.get_task(task_id)

    ### ======================= start evaluation ============================

    # 1. evaluate dataset loss
    try:
        dataset, shape_meta = get_dataset(
            dataset_path=os.path.join(
                cfg.folder, benchmark.get_task_demonstration(task_id)
            ),
            obs_modality=cfg.data.obs.modality,
            initialize_obs_utils=True,
            seq_len=cfg.data.seq_len,
        )
        dataset = GroupedTaskDataset(
            [dataset], task_embs[task_id : task_id + 1]
        )
    except:
        print(
            f"[error] failed to load task {task_id} name {benchmark.get_task_names()[task_id]}"
        )
        sys.exit(0)

    algo.eval()

    test_loss = 0.0

    # 2. evaluate success rate
    if alg == "multitask":
        save_folder = os.path.join(
            save_dir,
            f"{benchmark_name}_{alg}_{policy}_{seed}_ep{ep}_on{task_id}.stats",
        )
    else:
        save_folder = os.path.join(
            save_dir,
            f"{benchmark_name}_{alg}_{policy}_{seed}_load{load_task}_ep{ep}_on{task_id}.stats",
        )

    video_path = os.path.join(
        save_dir,
        "videos"
    )

    with Timer() as t, VideoWriter(
        video_path=video_path, 
        name=f"{benchmark_name}_{alg}_{policy}_{seed}_load{load_task}_ep{ep}_on{task_id}.mp4", 
        save_video=save_videos ) as video_writer:
        
        env_args = {
            "bddl_file_name": os.path.join(
                cfg.bddl_folder, task.problem_folder, task.bddl_file
            ),
            "camera_heights": cfg.data.img_h,
            "camera_widths": cfg.data.img_w,
        }

        env_num = 20
        env = SubprocVectorEnv(
            [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
        )
        env.reset()
        env.seed(cfg.seed)
        algo.reset()

        init_states_path = os.path.join(
            cfg.init_states_folder, task.problem_folder, task.init_states_file
        )
        init_states = torch.load(init_states_path)
        indices = np.arange(env_num) % init_states.shape[0]
        init_states_ = init_states[indices]

        dones = [False] * env_num
        steps = 0
        obs = env.set_init_state(init_states_)
        task_emb = benchmark.get_task_emb(task_id)

        num_success = 0
        for _ in range(EVAL_EVERY):  # simulate the physics without any actions
            env.step(np.zeros((env_num, 7)))

        with torch.no_grad():
            while steps < cfg.eval.max_steps:
                steps += 1

                data = raw_obs_to_tensor_obs(obs, task_emb, cfg)
                actions = algo.policy.get_action(data)
                obs, reward, done, info = env.step(actions)
                video_writer.append_vector_obs(
                    obs, dones, camera_name="agentview_image"
                )

                # check whether succeed
                for k in range(env_num):
                    dones[k] = dones[k] or done[k]
                if all(dones):
                    break

            for k in range(env_num):
                num_success += int(dones[k])

        success_rate = num_success / env_num
        env.close()

        # eval_stats = {
        #     "success_rate": success_rate,
        # }

        # os.system(f"mkdir -p {save_dir}")
        # torch.save(eval_stats, save_folder)
    # print_green(
    #     f"[info] finish for ckpt at {run_folder} in {t.get_elapsed_time()} sec for rollouts"
    # )
    print_green(f"[info] Videos are saved at {video_path}")
    print_green(f"[info] Success rate: {success_rate}")
    return success_rate

def print_green(text):
    """输出绿色文本"""
    GREEN = '\033[32m'  # 绿色
    RESET = '\033[0m'   # 重置颜色
    print(f"{GREEN}{text}{RESET}")

def all_eval():
    
    result = {}
    for load_id in range(N_TASKS):

        save_dir = os.path.join(save_path, "all_eval", f"model_{load_id}")
        
        os.system(f"mkdir -p {save_dir}")

        result[f"model_{load_id}"] = {}

        for ep in range(11):
            test_id = load_id
            start_time = time.time()
            print_green(f"start evaluation on model {load_id}, epoch {EVAL_EVERY*ep}, evaluated on {test_id}")
            success_rate = eval_one_task(
                run_folder = run_folder,
                benchmark_name=args.benchmark, 
                policy=args.policy, 
                alg=args.alg, 
                seed=args.seed, 
                load_task=load_id, 
                ep=EVAL_EVERY*ep, 
                task_id=test_id, 
                save_dir=save_dir,
                save_videos=args.save_videos,
                device_id=args.device_id
            )
            result[f"model_{load_id}"][f"epoch_{EVAL_EVERY*ep}"] = success_rate
            end_time = time.time()
            print_green(f"finish evaluation in time {end_time - start_time} seconds")
        
        save_folder = os.path.join(
            save_dir,
            f"result.stats",
        )


        #model_0: 0.6, 0.6, 0.55, 0.45, 0.6, 0.55, 0.3, 0.05, 0.0, 0.0, 0.0
        #model_1: 0.6, 0.5, 0.7, 0.35, 0.6, 0.4, 0.3, 0.55, 0.55, 0.1, 0.0
        
        # torch.save(result[f"model_{load_id}"], os.path.join(save_dir, f"result_model_{load_id}.stats",))
    
    succ = np.zeros((N_TASKS, N_EP))
    for load_id in range(N_TASKS):
        for ep in range(N_EP):
            succ[load_id][ep] = result[f"model_{load_id}"][f"epoch_{EVAL_EVERY*ep}"]
    torch.save(succ, os.path.join(save_path, "all_eval", "result.stats"))

    print_green("Finish evaluation")

def find_best():
    best_ep, best = {}, {}
    for id in range(N_TASKS):
        result_path = os.path.join(save_path, f"model_{id}", f"result_model_{id}.stats")
        result = torch.load(result_path)
        best_ep[id], best[id] = 0, result["epoch_0"]
        for ep in range(1, 11):
            if(result[f"epoch_{EVAL_EVERY*ep}"] > best[id]):
                best[id] = result[f"epoch_{EVAL_EVERY*ep}"]
                best_ep[id] = EVAL_EVERY*ep
        print_green(f"In model {id}, epoch {best_ep[id]} is the best with success rate {best[id]}")
    torch.save(best_ep, os.path.join(save_path, "best_ep.stats"))


def cross_eval():
    best_ep = torch.load(os.path.join(save_path, "best_ep.stats"))
    result = {}
    for load_id in range(N_TASKS):
        
        save_dir = os.path.join(save_path, "cross_eval", f"model_{load_id}")
    
        os.system(f"mkdir -p {save_dir}")

        result[f"model_{load_id}"] = {}

        for test_id in range(N_TASKS):

            start_time = time.time()
            print_green(f"start evaluation on model {load_id}, best epoch {best_ep[load_id]}, evaluated on {test_id}")
            success_rate = eval_one_task(
                run_folder=run_folder,
                benchmark_name=args.benchmark, 
                policy=args.policy, 
                alg=args.alg, 
                seed=args.seed, 
                load_task=load_id, 
                ep=best_ep[load_id], 
                task_id=test_id, 
                save_dir=save_dir,
                save_videos=args.save_videos,
                device_id=args.device_id
            )
            result[f"model_{load_id}"][f"task_{test_id}"] = success_rate
            end_time = time.time()
            print_green(f"finish evaluation in time {end_time - start_time} seconds")
            
        save_folder = os.path.join(
            save_dir,
            f"result.stats",
        )


        #model_0: 0.6, 0.6, 0.55, 0.45, 0.6, 0.55, 0.3, 0.05, 0.0, 0.0, 0.0
        #model_1: 0.6, 0.EVAL_EVERY, 0.7, 0.35, 0.6, 0.4, 0.3, 0.55, 0.55, 0.1, 0.0
        
        torch.save(result[f"model_{load_id}"], os.path.join(save_dir, f"result_model_{load_id}.stats"))
    succ = np.zeros((N_TASKS, N_TASKS))
    for load_id in range(N_TASKS):
        for task_id in range(N_TASKS):
            succ[load_id][task_id] = result[f"model_{load_id}"][f"task_{task_id}"]
    torch.save(succ, os.path.join(save_path, "cross_eval", "result.stats"))

    print_green("Finish evaluation")

def analysis():
    best_ep = torch.load(os.path.join(save_path, "best_ep.stats"))
    result = {}
    result["cross"] = torch.load(os.path.join(save_path, "cross_eval", "result.stats"))
    result["all"] = torch.load(os.path.join(save_path, "all_eval", "result.stats"))

    
    for i in range(N_TASKS):
        best = best_ep[i] // EVAL_EVERY
        result["all"][i][best:] = result["all"][i][best]

    fwt = np.mean(result["all"])

    bwt = 0.0
    auc = 0.0
    for k in range(N_TASKS):
        bwt_k = 0.0
        auc_k = 0.0
        for tau in range(k + 1, N_TASKS):
            bwt_k += result["cross"][k][k] - result["cross"][tau][k]
            auc_k += result["cross"][tau][k]
        if k + 1 < N_TASKS:
            bwt_k /= (N_TASKS - k - 1)
        auc_k = (auc_k + result["all"][k].mean()) / (N_TASKS - k)

        bwt += bwt_k
        auc += auc_k
    
    bwt /= N_TASKS
    auc /= N_TASKS

    result_summary = {
        "FWT": fwt,
        "NBT": bwt,
        "AUC": auc
    }

    save_file = os.path.join(save_path, "summary.stats")
    torch.save(result_summary, save_file)
    print_green(result_summary)
    print_green(f"Saved in {save_file}")


    # torch.save(result, os.path.join(save_path, "cross_eval", "result.stats"))

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for experiment parameters")

    parser.add_argument("--benchmark", type=str, default="libero_spatial", help="Benchmark name")
    parser.add_argument("--alg", type=str, default="er", help="Algorithm name")
    parser.add_argument("--policy", type=str, default="bc_vilt_policy", help="Policy type")
    parser.add_argument("--seed", type=int, default=10000, help="Random seed for experiment (default: 42)")
    parser.add_argument("--run-id", type=int, default=1, help="ID for the run")

    parser.add_argument("--save-videos", action="store_true", help="Flag to save videos")

    parser.add_argument("--device-id", type=int, default=0, help="ID of the device (default: 0)")
    
    parser.add_argument("--experiment-dir", type=str, default="./experiments", help="Directory for the experiment")

    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()
    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[args.benchmark]()
    N_TASKS = benchmark_instance.get_num_tasks()
    # 解析命令行参数
    experiment_dir = os.path.join(
            args.experiment_dir,
            f"{benchmark_map[args.benchmark]}/"
            + f"{algo_map[args.alg]}/"
            + f"{policy_map[args.policy]}_seed{args.seed}",
        )
    run_folder = os.path.join(experiment_dir, f"run_{args.run_id:03d}")
    save_path = os.path.join(run_folder, "evaluation")
    all_eval()
    # find_best()
    # cross_eval()
    # analysis()
