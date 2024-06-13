import isaacgym
from dataclasses import dataclass, field
from typing import List

from isaacgym import gymapi
import os
import numpy as np
from copy import deepcopy
from hydra import compose, initialize_config_dir
from pathlib import Path

import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo import RolloutPolicy

from bidexhands.tasks.hand_base.vec_task import VecTaskPython
from bidexhands.tasks.shadow_hand_scissors import ShadowHandScissors
from bidexhands.tasks.shadow_hand_bottle_cap import ShadowHandBottleCap
from bidexhands.tasks.shadow_hand_switch import ShadowHandSwitch
from bidexhands.utils.config import parse_sim_params, load_cfg, retrieve_cfg

bidexenvs_root = os.getenv("BIDEXHANDS_ROOT", Path("~/ngc/DexterousHands/bidexhands").expanduser())

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
        camera_names=["fixed_camera"]
    )
    print(stats)

