
def check_integrity():
    # Check if all the task names and their bddl file names
    task_names = benchmark_instance.get_task_names()
    # print("The benchmark contains the following tasks:")
    for i in range(num_tasks):
        task_name = task_names[i]
        task = benchmark_instance.get_task(i)
        bddl_file = os.path.join(bddl_files_default_path, task.problem_folder, task.bddl_file)
        # print(f"\t {task_name}, detail definition stored in {bddl_file}")
        if not os.path.exists(bddl_file):
            print(colored(f"[error] bddl file {bddl_file} cannot be found. Check your paths", "red"))

    # Check if all the init states files exist for tasks
    task_names = benchmark_instance.get_task_names()
    # print("The benchmark contains the following tasks:")
    for i in range(num_tasks):
        task_name = task_names[i]
        task = benchmark_instance.get_task(i)
        init_states_path = os.path.join(init_states_default_path, task.problem_folder, task.init_states_file)
        if not os.path.exists(init_states_path):
            print(colored(f"[error] the init states {init_states_path} cannot be found. Check your paths", "red"))
    # print(f"An example of init file is named like this: {task.init_states_file}")


def visual_insitial():
    from libero.libero.envs import OffScreenRenderEnv
    from PIL import Image

    import torch
    import torchvision

    # task_id is the (task_id + 1)th task in the benchmark
    task_id = 0
    task = benchmark_instance.get_task(task_id)

    env_args = {
        "bddl_file_name": os.path.join(bddl_files_default_path, task.problem_folder, task.bddl_file),
        "camera_heights": 128,
        "camera_widths": 128
    }

    env = OffScreenRenderEnv(**env_args)


    init_states = benchmark_instance.get_task_init_states(task_id)

    # Fix random seeds for reproducibility
    env.seed(0)

    def make_grid(images, nrow=8, padding=2, normalize=False, pad_value=0):
        """Make a grid of images. Make sure images is a 4D tensor in the shape of (B x C x H x W)) or a list of torch tensors."""
        grid_image = torchvision.utils.make_grid(images, nrow=nrow, padding=padding, normalize=normalize, pad_value=pad_value).permute(1, 2, 0)
        return grid_image

    images = []
    env.reset()

    # print(len(init_states))

    for eval_index in range(len(init_states)):
        env.set_init_state(init_states[eval_index])
        
        # t = 5 if eval_index % 2 == 0 else 0
        t = 1

        for _ in range(t):
            obs, _, _, _ = env.step([1.] * 7)
        images.append(torch.from_numpy(obs["agentview_image"]).permute(2, 0, 1))

    # # images = torch.stack(images, dim=0).permute(0, 3, 1, 2)
    # print(images.shape)
    grid_image = make_grid(images, nrow=10, padding=2, pad_value=0)
    # display(Image.fromarray(grid_image.numpy()[::-1]))
    image = Image.fromarray(grid_image.numpy()[::-1])
    image.save("outputs/initial_image_1.png")
    env.close()


def download_datasets():
    import libero.libero.utils.download_utils as download_utils

    download_dir = get_libero_path("datasets")
    datasets = "libero_spatial" # Can specify "all", "libero_goal", "libero_spatial", "libero_object", "libero_100"

    libero_datasets_exist = download_utils.check_libero_dataset(download_dir=download_dir, dataset=datasets)

    if not libero_datasets_exist:
        download_utils.libero_dataset_download(download_dir=download_dir, datasets=datasets)

    


def visual():
    import h5py
    from libero.libero.utils.dataset_utils import get_dataset_info
    import imageio

    # Check if the demo files exist
    # demo_files = [os.path.join(datasets_default_path, benchmark_instance.get_task_demonstration(i)) for i in range(num_tasks)]
    # for demo_file in demo_files:
    #     if not os.path.exists(demo_file):
    #         print(colored(f"[error] demo file {demo_file} cannot be found. Check your paths", "red"))

    example_demo_file = 'outputs/datasets/new_dataset.hdf5'
    # Print the dataset info. We have a standalone script for doing the same thing available at `scripts/get_dataset_info.py`
    # get_dataset_info(example_demo_file)

    with h5py.File(example_demo_file, "r") as f:
        images = f["data/demo_0/obs/agentview_rgb"][()]

    # Specify the output file path
    output_video_path = os.path.join("outputs", "outputs_1.mp4")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    # Write video to the specified path
    video_writer = imageio.get_writer(output_video_path, fps=60)
    for image in images:
        video_writer.append_data(image[::-1])
    video_writer.close()

    # HTML("""
    #     <video width="640" height="480" controls>
    #         <source src="output.mp4" type="video/mp4">
    #     </video>
    #     <script>
    #         var video = document.getElementsByTagName('video')[0];
    #         video.playbackRate = 2.0; // Increase the playback speed to 2x
    #         </script>    
    # """)
    

def print_hdf5_structure(group, indent=0):
    # 递归打印 HDF5 文件或组的结构，包括每一层的 keys、attributes 和元数据。
    
    # :param group: 当前处理的 h5py 文件或组
    # :param indent: 用于格式化输出的缩进，默认值为0

    # 遍历当前组的所有 keys
    import h5py
    for key, obj in group.items():
        print(" " * indent + f"Key: {key}, Type: {type(obj)}")
        
        # 打印对象的 attributes (元数据)
        if isinstance(obj, h5py.Group) or isinstance(obj, h5py.Dataset):
            if obj.attrs:
                print(" " * (indent + 2) + "Attributes:")
                for attr_key, attr_value in obj.attrs.items():
                    print(" " * (indent + 4) + f"{attr_key}")
        
        # 如果是 Group 类型，递归调用该函数以处理下一层
    
    for key, obj in group.items():
        if isinstance(obj, h5py.Group):
            print(" " * (indent + 2) + "Entering Group " + key + "...")
            print_hdf5_structure(obj, indent + 4)  # 递归进入下一层 Group
            break

def read_data(h5_file_path):
    import h5py
    # 打开 HDF5 文件
    with h5py.File(h5_file_path, 'r') as file:
        # print_hdf5_structure(file)
        with open("./model_file.txt", "w", encoding="utf-8") as output:
            output.write(file["data"]["demo_0"].attrs["model_file"])
        pass



def get_auc():
# def get_auc(experiment_dir, bench, algo, policy):
    import numpy as np
    import torch
    # N_EP = cfg.train.n_epochs // cfg.eval.eval_every + 1
    N_EP = 11
    # fwds = np.zeros((N_TASKS, N_EP, N_SEEDS))

    # for task in range(N_TASKS):
    counter = 0
    # for k, seed in enumerate(seeds):
    name = '/home/jiangtao/tianxing/LIBERO-master/experiments/LIBERO_SPATIAL/ER/BCRNNPolicy_seed42/run_039/task0_auc.log'
    # try:
    succ = torch.load(name)# (n_epochs)
    # idx = succ.argmax()
    # succ[idx:] = succ[idx]
    # fwds[task, :, k] = succ
    print(succ)
    # except:
    #     print("Some errors when loading results")
        # continue
    # return fwds



if __name__ == '__main__':
    from libero.libero import benchmark, get_libero_path, set_libero_default_path
    import os
    from termcolor import colored

    benchmark_root_path = get_libero_path("benchmark_root")
    init_states_default_path = get_libero_path("init_states")
    datasets_default_path = get_libero_path("datasets")
    bddl_files_default_path = get_libero_path("bddl_files")
    # print("Default benchmark root path: ", benchmark_root_path)
    # print("Default dataset root path: ", datasets_default_path)
    # print("Default bddl files root path: ", bddl_files_default_path)



    benchmark_dict = benchmark.get_benchmark_dict()
    # print(benchmark_dict)



    # initialize a benchmark
    benchmark_instance = benchmark_dict["libero_spatial"]()
    num_tasks = benchmark_instance.get_num_tasks()
    # see how many tasks involved in the benchmark
    # print(f"{num_tasks} tasks in the benchmark {benchmark_instance.name}: ")


    # Load torch init files
    init_states = benchmark_instance.get_task_init_states(0)
    # Init states in the same (num_init_rollouts, num_simulation_states)
    # print(init_states.shape)


    # check_integrity()
    # visual_insitial()
    # download_datasets()
    visual()
    # read_data('./libero/datasets/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5')
    # get_auc()






# 附：数据集结构
'''
[info] using task orders [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Key: data, Type: <class 'h5py._hl.group.Group'>
  Attributes:
    bddl_file_name
    env_args
    env_name
    macros_image_convention
    num_demos
    problem_info
    tag
    total
  Entering Group data...
    Key: demo_0, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_1, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_10, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_11, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_12, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_13, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_14, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_15, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_16, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_17, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_18, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_19, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_2, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_20, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_21, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_22, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_23, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_24, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_25, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_26, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_27, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_28, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_29, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_3, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_30, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_31, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_32, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_33, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_34, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_35, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_36, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_37, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_38, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_39, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_4, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_40, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_41, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_42, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_43, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_44, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_45, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_46, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_47, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_48, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_49, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_5, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_6, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_7, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_8, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
    Key: demo_9, Type: <class 'h5py._hl.group.Group'>
      Attributes:
        init_state
        model_file
        num_samples
      Entering Group demo_0...
        Key: actions, Type: <class 'h5py._hl.dataset.Dataset'>
        Key: dones, Type: <class 'h5py._hl.dataset.Dataset'>
        Key: obs, Type: <class 'h5py._hl.group.Group'>
        Key: rewards, Type: <class 'h5py._hl.dataset.Dataset'>
        Key: robot_states, Type: <class 'h5py._hl.dataset.Dataset'>
        Key: states, Type: <class 'h5py._hl.dataset.Dataset'>
          Entering Group obs...
            Key: agentview_rgb, Type: <class 'h5py._hl.dataset.Dataset'>
            Key: ee_ori, Type: <class 'h5py._hl.dataset.Dataset'>
            Key: ee_pos, Type: <class 'h5py._hl.dataset.Dataset'>
            Key: ee_states, Type: <class 'h5py._hl.dataset.Dataset'>
            Key: eye_in_hand_rgb, Type: <class 'h5py._hl.dataset.Dataset'>
            Key: gripper_states, Type: <class 'h5py._hl.dataset.Dataset'>
            Key: joint_states, Type: <class 'h5py._hl.dataset.Dataset'>
'''