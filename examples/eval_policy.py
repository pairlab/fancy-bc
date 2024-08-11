try:
    import isaacgym
    import isaacgymenvs
except ImportError:
    print("Unable to import IsaacGymEnvs package.")

import os
import gym
import numpy as np
from copy import deepcopy
from hydra import compose, initialize_config_dir
from pathlib import Path
from robomimic.utils.vis_utils import make_model_img_feature_plot
from omegaconf import OmegaConf

import torch
import json
import h5py

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
from robomimic.algo import RolloutPolicy

from env_bidex import create_bidex_env 

plt_root = os.getenv("POLICY_LEARNING_TOOLKIT_ROOT", Path("../../policy_learning_toolkit/").expanduser())
igenvs_root = os.getenv("ISAACGYM_ROOT", Path("~/diff_manip/external/IsaacGymEnvs").expanduser())
bidexenvs_root = os.getenv("BIDEXHANDS_ROOT", Path("~/ngc/DexterousHands/bidexhands").expanduser())


def get_obs_dict(obs, obs_keys, env):
    obs_dict = {}
    if hasattr(env, 'task'):
        env_obs_dict = env.task.obs_dict
    elif hasattr(env, 'obs_dict'):
        env_obs_dict = env.obs_dict
    else:
        # handle myodex envs
        obs_dict = {"vec_obs": obs, "fixed_camera": env.get_camera_obs()["fixed_camera"]}
    for k in obs_keys:
        if "camera" in k and env_obs_dict[k].shape[3] == 3:
            obs_dict[k] = env_obs_dict[k].permute(0, 3, 1, 2)
            print(obs_dict[k].shape)
        else:
            obs_dict[k] = env_obs_dict[k]



def rollout(policy, env, horizon, render=False, video_writer=None, video_skip=5, camera_names=None, device="cuda"):
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
            obs_dict = get_obs_dict(obs, obs_keys, env)

            # get action from policy
            act = policy(ob=obs_dict, batched=True)

            # play action
            next_obs, r, done, info = env.step(torch.tensor(act, device=device, dtype=torch.float))

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


class MyoSuiteCameraWrapper(gym.Wrapper):
    def __init__(self, env, frame_size=(64, 64), camera_names=['hand_side_inter']):
        super().__init__(env)
        self.env = env
        self.frame_size = frame_size
        self.camera_names = camera_names
        self.camera_observation_space = spaces.Dict({
            cam_name: spaces.Box(0, 255, (*frame_size, 3), dtype=np.uint8)
            for cam_name in self.camera_names
        })
        self.has_rendered = False

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info.update(self.get_camera_obs())
        info['success'] = info['solved']
        return obs, reward, done, info

    def get_camera_obs(self):
        if not self.has_rendered: 
            self.env.sim.renderer.render_offscreen(width=5, height=5, camera_id=-1)
            self.has_rendered = True
            self.env.sim.renderer._scene_option.flags[mjtVisFlag.mjVIS_STATIC] = 0
        return {cam_name: self.env.sim.renderer.render_offscreen(
            width=self.frame_size[0], height=self.frame_size[1],
            camera_id=cam_name) for cam_name in self.camera_names
            }

MYOSUITE_TASKS = {
    'myo-reach': 'myoHandReachFixed-v0',
    'myo-reach-hard': 'myoHandReachRandom-v0',
    'myo-pose': 'myoHandPoseFixed-v0',
    'myo-pose-hard': 'myoHandPoseRandom-v0',
    'myo-obj-hold': 'myoHandObjHoldFixed-v0',
    'myo-obj-hold-hard': 'myoHandObjHoldRandom-v0',
    'myo-key-turn': 'myoHandKeyTurnFixed-v0',
    'myo-key-turn-hard': 'myoHandKeyTurnRandom-v0',
    'myo-pen-twirl': 'myoHandPenTwirlFixed-v0',
    'myo-pen-twirl-hard': 'myoHandPenTwirlRandom-v0',
}

def eval_myo(ckpt_path):

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=ckpt_path, 
        device=device, 
        verbose=True
    )
     
    config, _ = FileUtils.config_from_checkpoint(algo_name=ckpt_dict["algo_name"], ckpt_dict=ckpt_dict, verbose=False)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=config.train.data[0]["path"])
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=config.train.data[0]["path"],
        action_keys=config.train.action_keys,
        all_obs_keys=config.all_obs_keys,
        verbose=True
    )
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, 
        render=False, 
        render_offscreen=True,
        use_image_obs=shape_meta["use_images"],
    )
    env = EnvUtils.wrap_env_from_config(env, config=config)
    #env = MyoSuiteCameraWrapper(env)
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
        camera_names=["hand_camera"],
        device=env.device
    )
    print(stats)
    visualize_feature_layer(policy.policy, env, hdf5_path=hdf5_path)

def visualize_feature_layer(policy, env, obs_dict=None, cam_obs_keys=None, hdf5_path=None):
    data = h5py.File(hdf5_path, "r")
    obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
    if cam_obs_keys is None:
        cam_obs_keys = list(filter(lambda x: "camera" in x, list(obs_encoder.obs_nets.keys())))
    print(cam_obs_keys)

    breakpoint()
    input_image = data["data/demo_0/obs"][cam_obs_keys[0]]
    data_dict = {'obs': {cam_obs_key: data['data/demo_0/obs/fixed_camera'][:] for cam_obs_key in cam_obs_keys},
                'actions': data['data/demo_0/actions'][:]}
    tensor_data_dict = TensorUtils.to_device(TensorUtils.to_tensor(data_dict), policy.policy.device)
    input_dict = policy.policy.process_batch_for_training(tensor_data_dict)
    input_dict = policy.policy.postprocess_batch_for_training(input_dict, obs_normalization_stats=None)

    print(input_image.shape)
    breakpoint()
    image_encoder = obs_encoder.obs_nets[cam_obs_keys[0]]
    feature_maps_layer, softmax_layer = image_encoder.nets[0], image_encoder.nets[1]
    make_model_img_feature_plot(hdf5_path, "", input_image, feature_maps_layer, softmax_layer)

def main(args):
    if args.isaacgym:
        eval_isaacgym(args.ckpt_path)
    elif args.bidexhands:
        eval_bidexhands(args.task, args.ckpt_path, args.rlgames, args.hdf5_path)
    elif args.myodex:
        eval_myo(args.ckpt_path)

if __name__ == "__main__":
    import argparse
    script_parser = argparse.ArgumentParser()
    script_parser.add_argument("--ckpt_path", type=str, default="")
    script_parser.add_argument("--hdf5_path", type=str, default="")
    script_parser.add_argument("--isaacgym", action="store_true")
    script_parser.add_argument("--bidexhands", action="store_true")
    script_parser.add_argument("--myodex", action="store_true")
    script_parser.add_argument("--task", type=str, default="ShadowHandScissors")
    script_parser.add_argument("--rlgames","--rlg", action="store_true")
    main(script_parser.parse_args())

