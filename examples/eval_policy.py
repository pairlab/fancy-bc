import isaacgym
import isaacgymenvs
import numpy as np
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


plt_root = Path("../../policy_learning_toolkit/").expanduser()
igenvs_root = Path("~/diff_manip/external/IsaacGymEnvs").expanduser()


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
            for k in obs_keys:
                if "camera" in k:
                    obs_dict[k] = env.obs_dict[k].permute(0, 3, 1, 2)
                    print(obs_dict[k].shape)
                else:
                    obs_dict[k] = env.obs_dict[k]

            # get action from policy
            act = policy(ob=obs_dict, batched=True)

            # play action
            next_obs, r, done, info = env.step(torch.tensor(act, device=env.device, dtype=torch.float))

            # compute reward
            total_reward += r
            success = info["success"].cpu().numpy().item()

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
            if done or success:
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
    dataset_path = cfg['train']['dataset']
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
    video_writer.close()

def eval_bidexhands(ckpt_path=None):


def main(args):
    if args.isaacgym:
        eval_isaacgym(args.ckpt_path)
    elif args.bidexhands:
        eval_bidexhands(args.ckpt_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--isaacgym", action="store_true")
    parser.add_argument("--bidexhands", action="store_true")
    main(parser.parse_args())

