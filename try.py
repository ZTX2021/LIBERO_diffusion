import argparse
import sys
import os

# TODO: find a better way for this?
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
from PIL import Image
from easydict import EasyDict
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoModel, pipeline, AutoTokenizer, logging
from pathlib import Path

from libero.libero import get_libero_path
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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("--experiment_dir", type=str, default="experiments")
    # for which task suite
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["libero_10", "libero_spatial", "libero_object", "libero_goal"],
    )
    parser.add_argument("--task_id", type=int, required=True)
    # method detail
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["base", "er", "ewc", "packnet", "multitask"],
    )
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        choices=["bc_rnn_policy", "bc_transformer_policy", "bc_vilt_policy"],
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ep", type=int)
    parser.add_argument("--load_task", type=int)
    parser.add_argument("--device_id", type=int)
    parser.add_argument("--save-videos", action="store_true")
    # parser.add_argument('--save_dir',  type=str, required=True)
    args = parser.parse_args()
    args.device_id = "cuda:" + str(args.device_id)
    args.save_dir = f"{args.experiment_dir}_saved"

    if args.algo == "multitask":
        assert args.ep in list(
            range(0, 50, 5)
        ), "[error] ep should be in [0, 5, ..., 50]"
    else:
        assert args.load_task in list(
            range(10)
        ), "[error] load_task should be in [0, ..., 9]"
    return args


if __name__ == "__main__":
    if multiprocessing.get_start_method(allow_none=True) != "spawn":  
        multiprocessing.set_start_method("spawn", force=True)
    sd, cfg, previous_mask = torch_load_model(
        model_path="/home/jiangtao/tianxing/LIBERO-master/experiments/LIBERO_SPATIAL/ER/BCRNNPolicy_seed42/run_039/task9_model_epoch50.pth", map_location="cuda:0"
    )
    # print(type(cfg))
    # pp = pprint.PrettyPrinter(indent=2)
    # pp.pprint(cfg)
    # print(type(sd))
    # print(sd)
    env_args = {
            "bddl_file_name": "/home/jiangtao/tianxing/LIBERO-master/libero/libero/./bddl_files/libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate.bddl",
            "camera_heights": 128,
            "camera_widths": 128,
        }
    env_num = 1
    env = SubprocVectorEnv(
        [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
    )
    env.reset()
    env.seed(42)
    init_states_path = os.path.join(
        "/home/jiangtao/tianxing/LIBERO-master/libero/libero/./init_files", "libero_spatial", "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate.pruned_init"
    )
    init_states = torch.load(init_states_path)
    # print(type(init_states))
    # print(init_states.shape)
    obs = env.set_init_state(init_states)[0]
    print(init_states.shape)
    print(obs['agentview_image'].shape)


    # 使用 PIL 创建图像
    image = Image.fromarray(obs['agentview_image'])

    # 保存图像
    image.save('output_image.png')