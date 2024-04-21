import isaacgym
from isaacgym import gymapi
import os
import isaacgym
import numpy as np
from copy import deepcopy
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pathlib import Path

import torch
import json
import h5py

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase
from robomimic.algo import RolloutPolicy
import imageio

try:
    from bidexhands.tasks.hand_base.vec_task import VecTaskPython
    from bidexhands.utils.config import parse_sim_params, load_cfg, retrieve_cfg
    from bidexhands.tasks.shadow_hand_scissors import ShadowHandScissors
    from bidexhands.tasks.shadow_hand_bottle_cap import ShadowHandBottleCap
    from bidexhands.tasks.shadow_hand_re_orientation import ShadowHandReOrientation
    from bidexhands.tasks.shadow_hand_switch import ShadowHandSwitch
    from bidexhands.utils.parse_task import parse_task
except:
    raise ImportError("bidexhands is not installed")

plt_root = os.getenv("POLICY_LEARNING_TOOLKIT_ROOT", Path("../../policy_learning_toolkit/").expanduser())
igenvs_root = os.getenv("ISAACGYM_ROOT", Path("~/diff_manip/external/IsaacGymEnvs").expanduser())
bidexenvs_root = os.getenv("BIDEXHANDS_ROOT", Path("~/ngc/DexterousHands/bidexhands").expanduser())


def rollout(policy, env, horizon, render=False, video_writer=None, video_skip=5, camera_names=None):
    """
    Helper function to carry out rollouts. Supports on-screen rendering, off-screen rendering to a video, 
    and returns the rollout trajectory.
    Args:
        policy (instance of RolloutPolicy): policy loaded from a checkpoint
        env (instance of EnvBase): env loaded from a checkpoint or demonstration metadata
        horizon (int): maximum horizon for the rollout
        render (bool): whether to render rollout on-screen
        video_writer (imageio writer): if provided, use to write rollout to video
        video_skip (int): how often to write video frames
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
    Returns:
        stats (dict): some statistics for the rollout - such as return, horizon, and task success
    """
    # assert isinstance(env, EnvBase)
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))

    policy.start_episode()
    obs = env.reset()
    obs_keys = set(policy.policy.nets.policy.obs_shapes.keys())
    # state_dict = env.get_state()

    # hack that is necessary for robosuite tasks for deterministic action playback
    # obs = env.reset_to(state_dict)

    results = {}
    video_count = 0  # video frame counter
    total_reward = 0.
    try:
        for step_i in range(horizon):
            obs_dict = {}
            if hasattr(env, 'task'):
                env_obs_dict = env.task.obs_dict
            else:
                env_obs_dict = env.obs_dict
            for k in obs_keys:
                if "camera" in k and env_obs_dict[k].shape[3] == 3:
                    obs_dict[k] = env_obs_dict[k].permute(0, 3, 1, 2)
                    print(obs_dict[k].shape)
                else:
                    obs_dict[k] = env_obs_dict[k]

            # get action from policy
            act = policy(ob=obs_dict, batched=True)

            # play action
            next_obs, r, done, info = env.step(torch.tensor(act, device=env.device, dtype=torch.float))

            # compute reward
            total_reward += r
            success_key = "success" if "success" in info else "successes"
            success = info[success_key].cpu().numpy()

            # visualization
            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(env.obs_dict[cam_name].cpu().numpy()[0])
                    video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                    video_writer.append_data(video_img)
                video_count += 1

            # break if done or if success
            if done.any() or success.any():
                break

            # update for next iter
            obs = deepcopy(next_obs)
            state_dict = env.get_state()

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))

    return stats

def eval_isaacgym(ckpt_path=None):

    ckpt_path = ckpt_path or "../bc_trained_models/test/20240403143734/models/model_epoch_2000.pth"
    ckpt_config = Path(ckpt_path).parent.parent / "config.json"
    with open(ckpt_config, "r") as f:
        cfg = json.load(f)
    dataset_path = cfg['train']['data'][0]['path']  # cfg['train']['dataset']
    with h5py.File(dataset_path, "r") as f:
        env_args = json.loads(f.attrs["env_args"])

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # restore policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)

    config_dir = str(igenvs_root / "isaacgymenvs" / "cfg")
    overrides = ["task=ArticulateTaskSprayScissorsCamera", "test=true", "num_envs=", 
                "train=ArticulateTaskPPONew"]
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg_expert = compose(config_name="config.yaml", overrides=overrides)
    # task_cfg = OmegaConf.load(igenvs_root / "task" / "ArticulateTaskSprayScissors.yaml")

    # OmegaConf.load("../../bc_trained_models/test/20240403143734/config.yaml")
    # ("runs/articulate_scissors1_relac_expert/config.yaml")
    env = isaacgymenvs.make(
                cfg_expert.seed,
                cfg_expert.task_name,
                cfg_expert.task.env.numEnvs,
                cfg_expert.sim_device,
                cfg_expert.rl_device,
                cfg_expert.graphics_device_id,
                cfg_expert.headless,
                cfg_expert.multi_gpu,
                cfg_expert.capture_video,
                cfg_expert.force_render,
                cfg_expert,
                # **kwargs,
            )
    env.rollout_exceptions = ()

    rollout_horizon = 150
    np.random.seed(0)
    torch.manual_seed(0)
    video_path = "rollout.mp4"
    # video_writer = imageio.get_writer(video_path, fps=20)

    stats = rollout(
        policy=policy, 
        env=env, 
        horizon=rollout_horizon, 
        render=True, 
        # video_writer=video_writer, 
        # video_skip=5, 
        camera_names=["hand_camera"]
    )
    print(stats)
    # video_writer.close()

from dataclasses import dataclass, field
from typing import List

@dataclass
class BidexEnvArgs:
    test: bool = False
    play: bool = False
    resume: int = 0
    checkpoint: str = "Base"
    headless: bool = False
    horovod: bool = False
    task: str = "ShadowHandScissors"
    task_type: str = "Python"
    rl_device: str = "cuda:0"
    logdir: str = "logs/"
    experiment: str = "Base"
    metadata: bool = False
    cfg_train: str = "Base"
    cfg_env: str = "Base"
    num_envs: int = 1
    episode_length: int = 0
    seed: int = field(default=None)
    max_iterations: int = -1
    steps_num: int = -1
    minibatch_size: int = -1
    randomize: bool = False
    torch_deterministic: bool = False
    algo: str = "ppo"
    model_dir: str = ""
    datatype: str = "random"
    sim_device: str = "cuda:0"
    device: str = "cuda"
    pipeline: str = "gpu"
    device_id: int = 0
    num_threads: int = 0
    subscenes: int = 0
    slices: int = 0
    headless: bool = True
    physics_engine: gymapi.SimType = gymapi.SIM_PHYSX
    use_rlg_config: bool = False

    def __post_init__(self):
        self.logdir, self.cfg_train, self.cfg_env = self.retrieve_cfg()
        self.use_gpu = self.sim_device.split(':')[0] in ["gpu", "cuda"]
        self.use_gpu_pipeline = self.pipeline == "gpu"

    def retrieve_cfg(self):
        _, cfg_train, cfg_env = retrieve_cfg(self, self.use_rlg_config)
        if self.logdir == "logs/":
            self.logdir = f"logs/{self.task}/{self.algo}"
        if self.cfg_train == "Base":
            self.cfg_train = cfg_train
        if self.cfg_env == "Base":
            self.cfg_env = cfg_env
        return self.logdir, self.cfg_train, self.cfg_env


def create_bidex_env(task, algo, use_rlgames):
    description = "Isaac Gym Example"
    headless = False
    os.chdir(bidexenvs_root)
    env_args = BidexEnvArgs(task=task, algo=algo, use_rlg_config=use_rlgames)

    cfg, cfg_train, logdir = load_cfg(env_args)
    sim_params = parse_sim_params(env_args, cfg, cfg_train)

    # create env 
    # def parse_task(args, cfg, cfg_train, sim_params, agent_index)
    task = eval(env_args.task)(
                cfg=cfg,
                sim_params=sim_params,
                physics_engine=env_args.physics_engine,
                device_type=env_args.device,
                device_id=env_args.device_id,
                headless=env_args.headless,
                is_multi_agent=False)
    env = VecTaskPython(task, rl_device=env_args.rl_device)
    return env

def eval_bidexhands(task, ckpt_path, rlgames, hdf5_path=None):
    # store cwd
    cwd = os.getcwd()
    env = create_bidex_env(task=task, algo="ppo", use_rlgames=rlgames)
    env.rollout_exceptions = ()
    env.device = env.task.device
    # change back to cwd
    os.chdir(cwd)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=env.task.device, verbose=True)

    rollout_horizon = 150 if task.endswith("Scissors") else 125
    stats = rollout(
        policy=policy, 
        env=env, 
        horizon=rollout_horizon, 
        render=False, 
        # video_writer=video_writer, 
        # video_skip=5, 
        camera_names=["hand_camera"]
    )
    print(stats)
    visualize_feature_layer(policy.policy, env, hdf5_path)

def visualize_feature_layer(policy, env, hdf5_path=None):
    data = h5py.File(hdf5_path, "r")
    cam_obs_keys = list(filter(lambda x: "camera" in x, list(data["data/demo_0/obs"].keys())))
    print(cam_obs_keys)
    breakpoint()
    input_image = data["data/demo_0/obs"][cam_obs_keys[0]]
    print(input_image.shape)
    breakpoint()
    image_encoder = policy.nets['policy'].nets['encoder'].nets['obs'].obs_nets[cam_obs_keys[0]]
    feature_maps_layer, softmax_layer = image_encoder.nets[0], image_encoder.nets[1]
    make_model_img_feature_plot(hdf5_path, "", input_image, feature_maps_layer, softmax_layer)

def main(args):
    if args.isaacgym:
        eval_isaacgym(args.ckpt_path)
    elif args.bidexhands:
        eval_bidexhands(args.task, args.ckpt_path, args.rlgames, args.hdf5_path)

if __name__ == "__main__":
    import argparse
    script_parser = argparse.ArgumentParser()
    script_parser.add_argument("--ckpt_path", type=str, default="")
    script_parser.add_argument("--hdf5_path", type=str, default="")
    script_parser.add_argument("--isaacgym", action="store_true")
    script_parser.add_argument("--bidexhands", action="store_true")
    script_parser.add_argument("--task", type=str, default="ShadowHandScissors")
    script_parser.add_argument("--rlgames","--rlg", action="store_true")
    main(script_parser.parse_args())

