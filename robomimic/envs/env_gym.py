"""
This file contains the gym environment wrapper that is used
to provide a standardized environment API for training policies and interacting
with metadata present in datasets.
"""
import json
import numpy as np
import os
from copy import deepcopy

import gym
try:
    import d4rl
except:
    print("WARNING: could not load d4rl environments!")

try:
    import myosuite
except:
    print("WARNING: could not load myosuite environments!")

os.environ["TDMPC_PATH"] = "/home/bsud/multi_task_experts/collect_myosuite/tdmpc2"
if os.environ.get("TDMPC_PATH") and os.path.exists(os.environ.get("TDMPC_PATH")):
    try:
        import sys
        sys.path.append(os.path.join(os.environ.get("TDMPC_PATH"), "tdmpc2"))
        from envs.myosuite import MyoSuiteWrapper
        print("Successfully imported MyoSuiteWrapper")
    except Exception as e:
        print("exception " , str(e))
        print("WARNING: could not load tdmpc2 environments!")
else:
    print("TDMPC path does not exist")

import robomimic.envs.env_base as EB
import robomimic.utils.obs_utils as ObsUtils


class EnvGym(EB.EnvBase):
    """Wrapper class for gym"""
    def __init__(
        self,
        env_name, 
        render=False, 
        render_offscreen=False, 
        use_image_obs=False, 
        postprocess_visual_obs=True, 
        **kwargs,
    ):
        """
        Args:
            env_name (str): name of environment. Only needs to be provided if making a different
                environment from the one in @env_meta.

            render (bool): ignored - gym envs always support on-screen rendering

            render_offscreen (bool): ignored - gym envs always support off-screen rendering

            use_image_obs (bool): ignored - gym envs don't typically use images

            postprocess_visual_obs (bool): ignored - gym envs don't typically use images
        """
        self._init_kwargs = deepcopy(kwargs)
        self._env_name = env_name
        self._current_obs = None
        self._current_reward = None
        self._current_done = None
        self._done = None
        self.env = gym.make(env_name, **kwargs)

    def step(self, action):
        """
        Step in the environment with an action.

        Args:
            action (np.array): action to take

        Returns:
            observation (dict): new observation dictionary
            reward (float): reward for this step
            done (bool): whether the task is done
            info (dict): extra information
        """
        obs, reward, done, info = self.env.step(action)
        self._current_obs = obs
        self._current_reward = reward
        self._current_done = done
        return self.get_observation(obs), reward, self.is_done(), info

    def reset(self):
        """
        Reset environment.

        Returns:
            observation (dict): initial observation dictionary.
        """
        self._current_obs = self.env.reset()
        self._current_reward = None
        self._current_done = None
        return self.get_observation(self._current_obs)

    def reset_to(self, state):
        """
        Reset to a specific simulator state.

        Args:
            state (dict): current simulator state that contains:
                - states (np.ndarray): initial state of the mujoco environment
        
        Returns:
            observation (dict): observation dictionary after setting the simulator state
        """
        if hasattr(self.env.unwrapped.sim, "set_state_from_flattened"):
            self.env.unwrapped.sim.set_state_from_flattened(state["states"])
            self.env.unwrapped.sim.forward()
            return { "flat" : self.env.unwrapped._get_obs() }
        else:
            raise NotImplementedError

    def render(self, mode="human", height=None, width=None, camera_name=None, **kwargs):
        """
        Render from simulation to either an on-screen window or off-screen to RGB array.

        Args:
            mode (str): pass "human" for on-screen rendering or "rgb_array" for off-screen rendering
            height (int): height of image to render - only used if mode is "rgb_array"
            width (int): width of image to render - only used if mode is "rgb_array"
        """
        if mode =="human":
            return self.env.render(mode=mode, **kwargs)
        if mode == "rgb_array":
            return self.env.render(mode="rgb_array", height=height, width=width)
        else:
            raise NotImplementedError("mode={} is not implemented".format(mode))

    def get_observation(self, obs=None):
        """
        Get current environment observation dictionary.

        Args:
            ob (np.array): current flat observation vector to wrap and provide as a dictionary.
                If not provided, uses self._current_obs.
        """
        if obs is None:
            assert self._current_obs is not None
            obs = self._current_obs
        return { "flat" : np.copy(obs) }

    def get_state(self):
        """
        Get current environment simulator state as a dictionary. Should be compatible with @reset_to.
        """
        # NOTE: assumes MuJoCo gym task!
        xml = self.env.sim.model.get_xml() # model xml file
        state = np.array(self.env.sim.get_state().flatten()) # simulator state
        return dict(model=xml, states=state)

    def get_reward(self):
        """
        Get current reward.
        """
        assert self._current_reward is not None
        return self._current_reward

    def get_goal(self):
        """
        Get goal observation. Not all environments support this.
        """
        raise NotImplementedError

    def set_goal(self, **kwargs):
        """
        Set goal observation with external specification. Not all environments support this.
        """
        raise NotImplementedError

    def is_done(self):
        """
        Check if the task is done (not necessarily successful).
        """
        assert self._current_done is not None
        return self._current_done

    def is_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """
        if hasattr(self.env.unwrapped, "_check_success"):
            return self.env.unwrapped._check_success()

        # gym envs generally don't check task success - we only compare returns
        return { "task" : False }

    @property
    def action_dimension(self):
        """
        Returns dimension of actions (int).
        """
        return self.env.action_space.shape[0]

    @property
    def name(self):
        """
        Returns name of environment name (str).
        """
        return self._env_name

    @property
    def type(self):
        """
        Returns environment type (int) for this kind of environment.
        This helps identify this env class.
        """
        return EB.EnvType.GYM_TYPE

    def serialize(self):
        """
        Save all information needed to re-instantiate this environment in a dictionary.
        This is the same as @env_meta - environment metadata stored in hdf5 datasets,
        and used in utils/env_utils.py.
        """
        return dict(env_name=self.name, type=self.type, env_kwargs=deepcopy(self._init_kwargs))

    @classmethod
    def create_for_data_processing(cls, env_name, camera_names, camera_height, camera_width, reward_shaping, **kwargs):
        """
        Create environment for processing datasets, which includes extracting
        observations, labeling dense / sparse rewards, and annotating dones in
        transitions. For gym environments, input arguments (other than @env_name)
        are ignored, since environments are mostly pre-configured.

        Args:
            env_name (str): name of gym environment to create

        Returns:
            env (EnvGym instance)
        """

        # make sure to initialize obs utils so it knows which modalities are image modalities.
        # For currently supported gym tasks, there are no image observations.
        obs_modality_specs = {
            "obs": {
                "low_dim": ["flat"],
                "rgb": [],
            }
        }
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)

        return cls(env_name=env_name, **kwargs)

    @property
    def rollout_exceptions(self):
        """
        Return tuple of exceptions to except when doing rollouts. This is useful to ensure
        that the entire training run doesn't crash because of a bad policy that causes unstable
        simulation computations.
        """
        return ()

    def __repr__(self):
        """
        Pretty-print env description.
        """
        return self.name + "\n" + json.dumps(self._init_kwargs, sort_keys=True, indent=4)

class EnvMyo(EnvGym):
    """
    This class combines the EnvGym and MyoSuiteWrapper to create a MyoSuite environment
    that is compatible with the EnvGym format. It optionally supports observation padding.
    """
    def __init__(self, env_name, pad_to_shape=None, **env_args):
        """
        Initialize the MyoSuite environment wrapped by EnvGym functionalities. Optionally pad observations.

        Args:
            env_name (str): name of the MyoSuite environment.
            pad_to_shape (tuple, optional): If provided, observations will be padded to this shape.
        """
        super().__init__(env_name)
        self.env = MyoSuiteWrapper(self.env, env_args)
        self._current_obs = None
        self._current_reward = None
        self._current_done = None
        self.pad_to_shape = pad_to_shape
        if pad_to_shape is not None:
            assert isinstance(pad_to_shape, tuple), "pad_to_shape must be a tuple"
            self.padding = [pad - s for pad, s in zip(pad_to_shape, self.env.observation_space.shape)]
            self.observation_space = gym.spaces.Dict({
                "vec_obs": gym.spaces.Box(
                    low=np.pad(self.env.observation_space.low, pad_width=[(0, pad) for pad in self.padding]),
                    high=np.pad(self.env.observation_space.high, pad_width=[(0, pad) for pad in self.padding])
                ),
                "fixed_image": gym.spaces.Box(
                    low=0, high=255, shape=(3, 64, 64), dtype=np.uint8  # Assuming fixed image size and RGB channels
                )
            })

    def render(self, **kwargs):
        """
        Render the environment using MyoSuite-specific rendering.

        Args:
            kwargs (dict): additional arguments to pass to the render method of the MyoSuiteWrapper.

        Returns:
            Rendered image or None.
        """
        return self.env.render(**kwargs)

    def get_observation(self, obs=None):
        """
        Get current environment observation dictionary. Optionally pad the observation if pad_to_shape is provided.

        Args:
            obs (np.array): current flat observation vector to wrap and provide as a dictionary.
                If not provided, uses self._current_obs.

        Returns:
            dict: observation dictionary
        """
        if obs is None:
            assert self._current_obs is not None
            obs = self._current_obs
        if self.pad_to_shape is not None:
            obs = np.pad(obs, [(0, pad) for pad in self.padding], mode='constant')
        return {"vec_obs": np.copy(obs), "fixed_camera": self.render(mode="rgb_array")}

    def __repr__(self):
        """
        Pretty-print env description.
        """
        return f"EnvMyo({self._env_name})\n" + json.dumps(self._init_kwargs, sort_keys=True, indent=4)


