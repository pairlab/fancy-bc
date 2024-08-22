"""
Example script for training a DAgger BC-VAE agent using the DaggerBC_VAE class.
This script is a modified version of train_bc_rnn.py, adapted for DAgger training.

To run a training session, use the following command:

    python train_dagger.py --dataset /path/to/dataset.hdf5 --output /path/to/output_dir
"""
import argparse
import os
import json
import robomimic
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.test_utils as TestUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.macros as Macros
import hydra
from robomimic.config import config_factory
from robomimic.scripts.train import train
from robomimic.algo import RolloutPolicy
from robomimic.envs.env_base import EnvBase
from robomimic.algo.distillation import DaggerBC_VAE

def get_config(dataset_path=None, output_dir=None):
    """
    Construct config for training.

    Args:
        dataset_path (str or None): path to hdf5 dataset
        output_dir (str): path to output folder
    """
    # make default BC config
    config = config_factory(algo_name="dagger")

    # set config attributes specific to DAgger
    config.algo_name = "dagger"
    config.experiment.num_epochs = 50
    config.experiment.rollout.beta.schedule_type = "linear"
    config.experiment.rollout.beta.start = 1.0
    config.experiment.rollout.beta.end = 0.0
    config.experiment.rollout.horizon = 400
    config.experiment.rollout.num_rollouts = 50

    # set dataset and output directory
    config.train.data = dataset_path
    config.train.output_dir = output_dir

    # set experiment config
    config.experiment.name = "dagger_bc_vae_example"
    config.experiment.validate = True
    config.experiment.logging.terminal_output_to_txt = True
    config.experiment.logging.log_tb = True

    # set train config
    config.train.num_data_workers = 0
    config.train.hdf5_cache_mode = "low_dim"
    config.train.hdf5_use_swmr = True
    config.train.hdf5_normalize_obs = False
    config.train.hdf5_filter_key = "train"
    config.train.hdf5_validation_filter_key = "valid"
    config.train.seq_length = 10
    config.train.dataset_keys = ("actions", "rewards", "dones")
    config.train.goal_mode = None
    config.train.cuda = True
    config.train.batch_size = 100
    config.train.num_epochs = 50

    return config


def load_policy(env, checkpoint_file_path, expert_cfg):
    from pathlib import Path
    from gym import spaces
    from isaacgymenvs.utils.reformat import omegaconf_to_dict
    from rl_games.algos_torch.network_builder import A2CBuilder
    from omegaconf import OmegaConf
    from isaacgymenvs.learning import actor_network_builder
    from rl_games.common import env_configurations
    from rl_games.algos_torch import players, model_builder

    model_builder.register_network("actor_critic_dict", lambda **kwargs: actor_network_builder.A2CBuilder())

    env_info = env_configurations.get_env_info(env.env)

    env_info = env_configurations.get_env_info(env.env)
    params = expert_cfg["train"]["params"]
    params["config"]["env_info"] = env_info
    algo = params["algo"]["name"]
    if algo == "a2c_continuous":
        agent = players.PpoPlayerContinuous(params=params)
        agent.restore(checkpoint_file_path)
        return agent
    params = omegaconf_to_dict(expert_cfg["train"]["params"])
    params["config"]["env_info"] = env_info
    algo = params["algo"]["name"]
    return agent


def main(args):
    # set torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # load config
    if args.config is not None:
        ext_cfg = json.load(open(args.config, 'r'))
        config = config_factory(ext_cfg["algo_name"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        with config.values_unlocked():
            config.update(ext_cfg)
    else:
        # Otherwise, use default config
        config = get_config(dataset_path=args.dataset, output_dir=args.output)

    # load training environment
    env_meta = FileUtils.get_env_metadata_from_dataset(config.train.data)
    env = EnvUtils.create_env_from_metadata(env_meta=env_meta)

    # create DaggerBC_VAE instance
    bc_vae = DaggerBC_VAE(
        config,
        obs_key_shapes=env.observation_spec(),
        ac_dim=env.action_spec()["action"][0],
        device=device,
    )

    policy = load_policy(env, args.expert_policy, args.expert_config)
    # load expert policy
    expert_policy = RolloutPolicy(
        policy,
        env=env,
        device=device,
        verbose=True,
    )

    # set expert policy in DaggerBC_VAE
    bc_vae.set_expert_policy(expert_policy)

    # run training
    train(config, bc_vae, device=device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Dataset path
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to input hdf5 dataset",
    )

    # Output dir
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="path to folder to use (or create) to output logs, model checkpoints, and rollout videos",
    )

    # Expert policy path
    parser.add_argument(
        "--expert_policy",
        type=str,
        required=True,
        help="path to expert policy checkpoint",
    )

    parser.add_argument("--config", type=str, help="path to config")
    parser.add_argument("--expert_config", type=str, help="path to expert config")

    args = parser.parse_args()

    # create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    main(args)
