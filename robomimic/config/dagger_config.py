from robomimic.config.bc_config import BCConfig

class DaggerConfig(BCConfig):
    ALGO_NAME = "dagger"

    def __init__(self):
        super(DaggerConfig, self).__init__()

    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the 
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config` 
        argument to the constructor. Any parameter that an algorithm needs to determine its 
        training and test-time behavior should be populated here.
        """
        super(DaggerConfig, self).algo_config()

        algo_cfg = self.algo

        # Dagger specific parameters
        self.train.seed = 0
        algo_cfg.collect_expert_dataset = False
        self.train.num_epochs = 10
        self.train.num_iterations_per_epoch = 100
        algo_cfg.num_warmup_iterations = 0
        self.train.batch_size = 64
        algo_cfg.num_batches = -1
        self.train.seq_length = 1
        algo_cfg.num_valid_batches = 50
        algo_cfg.num_eval_games = 100

        # Rollout parameters
        algo_cfg.rollout.enabled = False
        algo_cfg.rollout.num_rollouts = 100
        algo_cfg.rollout.max_episode_length = 1000
        algo_cfg.rollout.beta.schedule = "linear"
        algo_cfg.rollout.beta.beta_start = 1.0
        algo_cfg.rollout.beta.beta_end = 0.0

        # Optimizer parameters
        algo_cfg.optim_params.policy.optimizer_type = "adam"
        algo_cfg.optim_params.policy.learning_rate.initial = 1e-4
        algo_cfg.optim_params.policy.learning_rate.decay_factor = 0.1
        algo_cfg.optim_params.policy.learning_rate.epoch_schedule = []
        algo_cfg.optim_params.policy.learning_rate.scheduler_type = "linear"

        # Dataset parameters
        self.train.hdf5_cache_mode = "all"

    def train_config(self):
        """
        Dagger algorithms don't need "next_obs" from hdf5 - so save on storage and compute by disabling it.
        """
        super(DaggerConfig, self).train_config()
        self.train.hdf5_load_next_obs = False
