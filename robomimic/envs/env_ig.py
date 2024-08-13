"""
Wrapper environment class to enable using Isaacgymenvs-based environments
"""

from copy import deepcopy
import numpy as np
import json
from omegaconf import OmegaConf
from pathlib import Path
import hydra
import gym
import isaacgymenvs
import tempfile
import os
import yaml
import cv2

import robomimic.utils.obs_utils as ObsUtils
import robomimic.envs.env_base as EB


class EnvIsaacGym(EB.EnvBase):
    """
    Wrapper class for isaacgymenvs environments (mainly object joint articulation)
    """
    def __init__(
            self,
            env_name,
            ig_config,
            postprocess_visual_obs=True,
            render=False,
            render_offscreen=False,
            use_image_obs=False,
            lang=None,
            image_height=None,
            image_width=None,
            physics_timestep=1./240.,
            action_timestep=1./20.,
            **kwargs,
    ):
        """
        Args:
            ig_config (dict): YAML configuration to use for iGibson, as a dict

            postprocess_visual_obs (bool): if True, postprocess image observations
                to prepare for learning

            render (bool): if True, environment supports on-screen rendering

            render_offscreen (bool): if True, environment supports off-screen rendering. This
                is forced to be True if @use_image_obs is True.

            use_image_obs (bool): if True, environment is expected to render rgb image observations
                on every env.step call. Set this to False for efficiency reasons, if image
                observations are not required.

            render_mode (str): How to run simulation rendering. Options are {"pbgui", "iggui", or "headless"}

            image_height (int): If specified, overrides internal iG image height when rendering

            image_width (int): If specified, overrides internal iG image width when rendering

            physics_timestep (float): Pybullet physics timestep to use

            action_timestep (float): Action timestep to use for robot in simulation

            kwargs (unrolled dict): Any args to substitute in the ig_configuration
        """
        self._env_name = env_name
        if isinstance(ig_config, str):
            cfg = OmegaConf.to_container(OmegaConf.load(ig_config), resolve=True)
        elif isinstance(ig_config, dict):
            ig_config_overrides = list(map(lambda x: f"{x}={ig_config[x]}", ig_config.keys()))
            with hydra.initialize_config_dir(str(Path(isaacgymenvs.__path__[0]) / "cfg")):
                cfg = hydra.compose(config_name="config.yaml", overrides=ig_config_overrides)
        assert use_image_obs == cfg["task"]["env"].get("enableCameraSensors", False), "use_image_obs and ig_config enableCameraSensors must match"
        assert env_name == cfg["task"]["name"], "env_name and ig_config task name must match"
        self.ig_config = deepcopy(cfg)
        self.postprocess_visual_obs = postprocess_visual_obs
        self._init_kwargs = kwargs

        # Determine rendering mode
        self.render_mode = self.ig_config["render_mode"]
        self.render_onscreen = render

        # Make sure rgb is part of obs in ig config
        self.ig_config["output"] = list(set(self.ig_config["output"] + ["rgb"]))

        # Update ig config
        for k, v in kwargs.items():
            assert k in self.ig_config, f"Got unknown ig configuration key {k}!"
            self.ig_config[k] = v

        # Set rendering values
        self.obs_img_height = image_height if image_height is not None else self.ig_config["env"].get("camera_spec", {"height": 120})["height"]
        self.obs_img_width = image_width if image_width is not None else self.ig_config["env"].get("camera_spec", {"width": 120})["width"]

        # Create environment
        self.env = isaacgymenvs.make(
            self.ig_config["seed"],
            self._env_name,
            self.ig_config["env"]["numEnvs"],
            self.ig_config["sim_device"],
            self.ig_config["rl_device"],
            self.ig_config["graphics_device_id"],
            self.ig_config["headless"],
            self.ig_config["multi_gpu"],
            self.ig_config["capture_video"],
            self.ig_config["force_render"],
            self.ig_config,
            **self._init_kwargs
        )

        if self.ig_config["capture_video"]:
            self.env.is_vector_env = True
            self.env = gym.wrappers.RecordVideo(
                self.env,
                f"videos/{self._env_name}",
                step_trigger=lambda step: step % self.ig_config["capture_video_freq"] == 0,
                video_length=self.ig_config["capture_video_len"],
            )


    def step(self, action):
        """
        Step in the environment with an action

        Args:
            action: action to take

        Returns:
            observation: new observation
            reward: step reward
            done: whether the task is done
            info: extra information
        """
        obs, r, done, info = self.env.step(action)
        obs = self.get_observation(obs)
        return obs, r, self.is_done(), info

    def reset(self):
        """Reset environment"""
        di = self.env.reset()
        return self.get_observation(di)

    def reset_to(self, state):
        """
        Reset to a specific state
        Args:
            state (dict): contains:
                - states (np.ndarray): initial state of the mujoco environment
                - goal (dict): goal components to reset
        Returns:
            new observation
        """
        if "states" in state:
            self.env.reset_to(state["states"], exclude=self.exclude_body_ids)

        if "goal" in state:
            self.set_goal(**state["goal"])

        # Return obs
        return self.get_observation()

    def render(self, mode="human", camera_name="rgb", height=None, width=None):
        """
        Render

        Args:
            mode (str): Mode(s) to render. Options are either 'human' (rendering onscreen) or 'rgb' (rendering to
                frames offscreen)
            camera_name (str): Name of the camera to use -- valid options are "rgb" or "rgb_wrist"
            height (int): If specified with width, resizes the rendered image to this height
            width (int): If specified with height, resizes the rendered image to this width

        Returns:
            array or None: If rendering to frame, returns the rendered frame. Otherwise, returns None
        """
        # Only robotview camera is currently supported
        assert camera_name in {"rgb", "rgb_wrist"}, \
            f"Only rgb, rgb_wrist cameras currently supported, got {camera_name}."

        if mode == "human":
            assert self.render_onscreen, "Rendering has not been enabled for onscreen!"
            self.env.simulator.sync()
        else:
            assert self.env.simulator.renderer is not None, "No renderer enabled for this env!"

            frame = self.env.sensors["vision"].get_obs(self.env)[camera_name]

            # Reshape all frames
            if height is not None and width is not None:
                frame = cv2.resize(frame, dsize=(height, width), interpolation=cv2.INTER_CUBIC)
                return frame

    def resize_obs_frame(self, frame):
        """
        Resizes frame to be internal height and width values
        """
        return cv2.resize(frame, dsize=(self.obs_img_width, self.obs_img_height), interpolation=cv2.INTER_CUBIC)

    def get_observation(self, di=None):
        """Get environment observation"""
        if di is None:
            di = self.env.obs_dict
        ret = di
        for k in ret:
            # RGB Images
            if "rgb" in k or k in list(self.ig_config["env"].get("camera_spec", {}).keys()):
                ret[k] = di[k]
                # ret[k] = np.transpose(di[k], (2, 0, 1))
                if self.postprocess_visual_obs:
                    ret[k] = ObsUtils.process_obs(obs=self.resize_obs_frame(ret[k]), obs_key=k)

            # Depth images
            elif "depth" in k:
                # ret[k] = np.transpose(di[k], (2, 0, 1))
                # Values can be corrupted (negative or > 1.0, so we clip values)
                ret[k] = np.clip(di[k], 0.0, 1.0)
                if self.postprocess_visual_obs:
                    ret[k] = ObsUtils.process_obs(obs=self.resize_obs_frame(ret[k])[..., None], obs_key=k)

            # Segmentation Images
            elif "seg" in k:
                ret[k] = di[k][..., None]
                if self.postprocess_visual_obs:
                    ret[k] = ObsUtils.process_obs(obs=self.resize_obs_frame(ret[k]), obs_key=k)

            # Scans
            elif "scan" in k:
                ret[k] = np.transpose(np.array(di[k]), axes=(1, 0))

        return ret

    def sync_task(self):
        """
        Method to synchronize iG task, since we're not actually resetting the env but instead setting states directly.
        Should only be called after resetting the initial state of an episode
        """
        self.env.task.update_target_object_init_pos()
        self.env.task.update_location_info()

    def set_task_conditions(self, task_conditions):
        """
        Method to override task conditions (e.g.: target object), useful in cases such as playing back
            from demonstrations

        Args:
            task_conditions (dict): Keyword-mapped arguments to pass to task instance to set internally
        """
        self.env.set_task_conditions(task_conditions)

    def get_state(self):
        """Get iG flattened state"""
        return self.env.obs_buf

    def get_reward(self):
        return self.env.rew_buf
        # return float(self.is_success()["task"])

    def get_goal(self):
        """Get goal specification"""
        return self.env.object_target_dof_pos 

    def set_goal(self, **kwargs):
        """Set env target with external specification"""
        raise NotImplementedError

    def is_done(self):
        """Check if the agent is done (not necessarily successful)."""
        return False

    def is_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """
        return { "task" : self.env.successes }

    @classmethod
    def create_for_data_processing(
            cls,
            env_name,
            camera_names,
            camera_height,
            camera_width,
            reward_shaping,
            **kwargs,
    ):
        """
        Create environment for processing datasets, which includes extracting
        observations, labeling dense / sparse rewards, and annotating dones in
        transitions.

        Args:
            env_name (str): name of environment
            camera_names (list of str): list of camera names that correspond to image observations
            camera_height (int): camera height for all cameras
            camera_width (int): camera width for all cameras
            reward_shaping (bool): if True, use shaped environment rewards, else use sparse task completion rewards
        """
        has_camera = (len(camera_names) > 0)

        # note that @postprocess_visual_obs is False since this env's images will be written to a dataset
        return cls(
            env_name=env_name,
            render=False,
            render_offscreen=has_camera,
            use_image_obs=has_camera,
            postprocess_visual_obs=False,
            image_height=camera_height,
            image_width=camera_width,
            **kwargs,
        )

    @property
    def action_dimension(self):
        """Action dimension"""
        return self.env.robots[0].action_dim

    @property
    def name(self):
        """Environment name"""
        return self._env_name

    @property
    def type(self):
        """Environment type"""
        return EB.EnvType.IGENVS_TYPE

    def serialize(self):
        """Serialize to dictionary"""
        return dict(env_name=self.name, type=self.type,
                    ig_config=self.ig_config,
                    env_kwargs=deepcopy(self._init_kwargs))

    @classmethod
    def deserialize(cls, info, postprocess_visual_obs=True):
        """Create environment with external info"""
        return cls(env_name=info["env_name"], ig_config=info["ig_config"], postprocess_visual_obs=postprocess_visual_obs, **info["env_kwargs"])

    @property
    def rollout_exceptions(self):
        """Return tuple of exceptions to except when doing rollouts"""
        return (RuntimeError)

    def __repr__(self):
        return self.name + "\n" + json.dumps(self._init_kwargs, sort_keys=True, indent=4) + \
               "\nIGEnvs Config: \n" + json.dumps(self.ig_config, sort_keys=True, indent=4)

    @property
    def version(self):
        """
        Returns version of isaacgymenvs used for this environment, eg. 1.5.1
        """
        import pkg_resources
        return pkg_resources.get_distribution("isaacgymenvs").version
