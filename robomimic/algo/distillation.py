from robomimic.algo.bc import BC_VAE
from robomimic.config.dagger_config import DaggerConfig
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
from robomimic.algo import RolloutPolicy
from robomimic.utils.dataset import SequenceDataset
import h5py
import os
import numpy as np
import torch
from tqdm import tqdm
from collections import OrderedDict


class DaggerBC_VAE(BC_VAE):
    def __init__(self, config, obs_key_shapes, ac_dim, device):
        super().__init__(config, obs_key_shapes, ac_dim, device)

        self.dagger_config = config.dagger
        self.beta_schedule = self.dagger_config.rollout.beta.schedule_type
        self.beta_start = self.dagger_config.rollout.beta.start
        self.beta_end = self.dagger_config.rollout.beta.end
        self.rollout_horizon = self.dagger_config.rollout.horizon
        self.num_rollouts = self.dagger_config.rollout.num_rollouts
        self.num_epochs = self.dagger_config.num_epochs

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training
        return self.nets["policy"](obs_dict, goal_dict=goal_dict)

    def get_beta(self, epoch):
        if self.beta_schedule == "linear":
            return self.beta_start + (self.beta_end - self.beta_start) * (
                epoch / self.num_epochs
            )
        elif self.beta_schedule == "exponential":
            return self.beta_start * (self.beta_end / self.beta_start) ** (
                epoch / self.num_epochs
            )
        else:
            return self.beta_start  # constant schedule

    def train(self, train_loader, valid_loader=None, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.num_epochs

        for epoch in range(num_epochs):
            # DAgger data collection
            beta = self.get_beta(epoch)
            new_data = self.collect_rollouts(beta)

            # Add new data to dataset
            self.add_to_dataset(new_data)

            # BC training
            super().train(train_loader, valid_loader, num_epochs=1)

    def collect_rollouts(self, beta):
        new_data = OrderedDict()
        new_data["obs"] = []
        new_data["actions"] = []

        for _ in tqdm(
            range(self.num_rollouts), desc=f"Collecting rollouts (beta={beta:.2f})"
        ):
            obs = self.env.reset()
            for t in range(self.rollout_horizon):
                action_learner = self.get_action(obs)
                action_expert = self.get_expert_action(obs)

                # DAgger mixing
                if np.random.random() < beta:
                    action = action_expert
                else:
                    action = action_learner

                new_data["obs"].append(obs)
                new_data["actions"].append(action_expert)  # Always store expert action

                obs, _, done, _ = self.env.step(action)
                if done:
                    break

        return new_data

    def add_to_dataset(self, new_data):
        # Implement logic to add new_data to your dataset
        # This might involve extending your train_loader or modifying the underlying dataset
        pass

    def set_expert_policy(self, expert_policy):
        """
        Set the expert policy for collecting expert actions during training.

        Args:
            expert_policy (RolloutPolicy or list): Either a single expert policy or a list of expert policies.
        """
        if isinstance(expert_policy, list):
            self.expert_policies = [
                RolloutPolicy(p) if not isinstance(p, RolloutPolicy) else p
                for p in expert_policy
            ]
        else:
            self.expert_policies = [
                RolloutPolicy(expert_policy)
                if not isinstance(expert_policy, RolloutPolicy)
                else expert_policy
            ]

        # Ensure all expert policies are on the correct device
        for policy in self.expert_policies:
            policy.policy.to(self.device)

    def get_expert_action(self, obs):
        """
        Get expert action for a given observation.

        If multiple expert policies are available, randomly select one.

        Args:
            obs (torch.Tensor): Current observation

        Returns:
            torch.Tensor: Expert action
        """
        if not hasattr(self, "expert_policies"):
            raise AttributeError("Expert policy not set. Call set_expert_policy first.")

        # Randomly select an expert policy if multiple are available
        expert_policy = self.rng.choice(self.expert_policies)

        # Ensure observation is on the correct device
        obs = TensorUtils.to_device(obs, self.device)

        # Get action from expert policy
        with torch.no_grad():
            action = expert_policy(obs)

        return action


class ExtendableSequenceDataset(SequenceDataset):
    def __init__(
        self,
        hdf5_path,
        obs_keys,
        action_keys,
        dataset_keys,
        action_config,
        frame_stack=1,
        seq_length=1,
        pad_frame_stack=True,
        pad_seq_length=True,
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode=None,
        hdf5_use_swmr=True,
        hdf5_normalize_obs=False,
        filter_by_attribute=None,
        load_next_obs=True,
        shuffled_obs_key_groups=None,
        lang=None,
        max_num_demos=5000,
        keep_original_dataset=False,
        new_hdf5_file_folder="/tmp",
        stride=1,
    ):
        super().__init__(
            hdf5_path,
            obs_keys,
            action_keys,
            dataset_keys,
            action_config,
            frame_stack,
            seq_length,
            pad_frame_stack,
            pad_seq_length,
            get_pad_mask,
            goal_mode,
            hdf5_cache_mode,
            hdf5_use_swmr,
            hdf5_normalize_obs,
            filter_by_attribute,
            load_next_obs,
            shuffled_obs_key_groups,
            lang,
        )
        
        self.max_num_demos = max_num_demos
        self.keep_original_dataset = keep_original_dataset
        self.new_hdf5_file_folder = new_hdf5_file_folder
        self.stride = stride
        self.new_hdf5_cnt = 0
        self.num_demos_in_original_dataset = self.n_demos

    def load(self, hdf5_dataset_paths):
        self.hdf5_files = []
        self.hdf5_paths = []
        
        for hdf5_path in hdf5_dataset_paths:
            if os.path.isdir(hdf5_path):
                hdf5_files = [
                    os.path.join(hdf5_path, f)
                    for f in os.listdir(hdf5_path)
                    if os.path.isfile(os.path.join(hdf5_path, f)) and f.endswith(".hdf5")
                ]
                hdf5_files.sort(key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]))
                self.hdf5_files.extend([h5py.File(hdf5_file, "r", swmr=True, libver="latest") for hdf5_file in hdf5_files])
                self.hdf5_paths.extend([None for _ in hdf5_files])
            else:
                self.hdf5_files.append(h5py.File(hdf5_path, "r", swmr=True, libver="latest"))
                self.hdf5_paths.append(hdf5_path)

        for hdf5_file in self.hdf5_files:
            self.load_demo_info(filter_by_attribute=self.filter_by_attribute, hdf5_file=hdf5_file)
        
        self.build_data_indexing()

    def load_demo_info(self, filter_by_attribute=None, demos=None, hdf5_file=None):
        """
        Args:
            filter_by_attribute (str): if provided, use the provided filter key
                to select a subset of demonstration trajectories to load

            demos (list): list of demonstration keys to load from the hdf5 file. If 
                omitted, all demos in the file (or under the @filter_by_attribute 
                filter key) are used.
        """
        if hdf5_file is None:
            hdf5_file = self.hdf5_file

        # filter demo trajectory by mask
        if demos is not None:
            if hasattr(self, 'demos'):
                self.demos.extend(demos)
            else:
                self.demos = demos
        elif filter_by_attribute is not None:
            new_demos = [elem.decode("utf-8") for elem in np.array(hdf5_file["mask/{}".format(filter_by_attribute)][:])]
            if hasattr(self, 'demos'):
                self.demos.extend(new_demos)
            else:
                self.demos = new_demos
        else:
            new_demos = list(hdf5_file["data"].keys())
            if hasattr(self, 'demos'):
                self.demos.extend(new_demos)
            else:
                self.demos = new_demos

        # sort demo keys
        inds = np.argsort([int(elem[5:]) for elem in self.demos])
        self.demos = [self.demos[i] for i in inds]

        self.n_demos = len(self.demos)

        # keep internal index maps to know which transitions belong to which demos
        if not hasattr(self, '_index_to_demo_id'):
            self._index_to_demo_id = dict()  # maps every index to a demo id
        if not hasattr(self, '_demo_id_to_start_indices'):
            self._demo_id_to_start_indices = dict()  # gives start index per demo id
        if not hasattr(self, '_demo_id_to_demo_length'):
            self._demo_id_to_demo_length = dict()
        if not hasattr(self, '_demo_id_to_hdf5_file'):
            self._demo_id_to_hdf5_file = dict()  # maps demo ranges to specific hdf5 files

        # determine index mapping
        if not hasattr(self, 'total_num_sequences'):
            self.total_num_sequences = 0

        # Get the start index of new demos
        start_index = len(self._demo_id_to_start_indices)

        for ep in self.demos[start_index:]:
            demo_length = hdf5_file["data/{}".format(ep)].attrs["num_samples"]
            self._demo_id_to_start_indices[ep] = self.total_num_sequences
            self._demo_id_to_demo_length[ep] = demo_length
            self._demo_id_to_hdf5_file[ep] = hdf5_file

            num_sequences = demo_length
            # determine actual number of sequences taking into account whether to pad for frame_stack and seq_length
            if not self.pad_frame_stack:
                num_sequences -= (self.n_frame_stack - 1)
            if not self.pad_seq_length:
                num_sequences -= (self.seq_length - 1)

            if self.pad_seq_length:
                assert demo_length >= 1  # sequence needs to have at least one sample
                num_sequences = max(num_sequences, 1)
            else:
                assert num_sequences >= 1  # assume demo_length >= (self.n_frame_stack - 1 + self.seq_length)

            for _ in range(num_sequences):
                self._index_to_demo_id[self.total_num_sequences] = ep
                self.total_num_sequences += 1

    def add(self, rollouts):
        dataset_path = os.path.join(self.new_hdf5_file_folder, f"{self.new_hdf5_cnt}.hdf5")
        data_writer = h5py.File(dataset_path, "w")
        data_grp = data_writer.create_group("data")

        num_rollouts = len(rollouts["actions"])
        for rollout_idx in range(num_rollouts):
            length = round(np.sum(rollouts["masks"][rollout_idx]))

            if length == 0:  # special case: avoid processing 0-length demos
                continue

            demo_name = f"demo_{self.n_demos}"
            ep_data_grp = data_grp.create_group(demo_name)
            ep_data_grp.create_dataset("actions", data=rollouts["actions"][rollout_idx, :length])
            for k in rollouts["obs"].keys():
                ep_data_grp.create_dataset(f"obs/{k}", data=rollouts["obs"][k][rollout_idx, :length])
            ep_data_grp.attrs["num_samples"] = length

            self.demos.append(demo_name)
            self._demo_id_to_start_indices[demo_name] = self.total_num_sequences
            self._demo_id_to_demo_length[demo_name] = length
            self.n_demos += 1

        self.new_hdf5_cnt += 1

        data_writer.flush()
        data_writer.close()

        self.hdf5_files.append(h5py.File(dataset_path, "a"))
        self.hdf5_paths.append(dataset_path)

        # Truncate dataset if necessary
        if self.n_demos > self.max_num_demos:
            self._truncate_dataset()

        self.build_data_indexing()

    def _truncate_dataset(self):
        num_demos_to_remove = self.n_demos - self.max_num_demos
        if self.keep_original_dataset:
            remove_start = self.num_demos_in_original_dataset
            remove_end = remove_start + num_demos_to_remove
        else:
            remove_start = 0
            remove_end = num_demos_to_remove

        for demo in self.demos[remove_start:remove_end]:
            del self._demo_id_to_start_indices[demo]
            del self._demo_id_to_demo_length[demo]

        del self.demos[remove_start:remove_end]

        # Delete datasets on disk
        for hdf5_idx, hdf5_path in enumerate(self.hdf5_paths[self.num_demos_in_original_dataset:]):
            if hdf5_path is not None:
                self.hdf5_files[hdf5_idx + self.num_demos_in_original_dataset].close()
                os.remove(hdf5_path)
                print(f"Deleting dataset {hdf5_path}")

        self.hdf5_files = self.hdf5_files[:self.num_demos_in_original_dataset] + self.hdf5_files[remove_end:]
        self.hdf5_paths = self.hdf5_paths[:self.num_demos_in_original_dataset] + self.hdf5_paths[remove_end:]

        self.n_demos = self.max_num_demos

    def build_data_indexing(self):
        self.total_num_sequences = 0
        self._index_to_demo_id = {}

        for demo in self.demos:
            demo_length = self._demo_id_to_demo_length[demo]
            self._demo_id_to_start_indices[demo] = self.total_num_sequences

            num_sequences = demo_length
            if not self.pad_frame_stack:
                num_sequences -= (self.n_frame_stack - 1)
            if not self.pad_seq_length:
                num_sequences -= (self.seq_length - 1)
