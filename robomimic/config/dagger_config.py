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

        # Dagger specific parameters
        self.train.seed = 0
        self.train.num_epochs = 10
        self.train.num_iterations_per_epoch = 100
        self.train.batch_size = 64
        self.train.seq_length = 1
        self.train.data_format = "extendable"

        # Rollout parameters
        self.experiment.rollout.enabled = True
        self.experiment.rollout.n = 100
        self.experiment.rollout.horizon = 500
        self.experiment.rollout.beta.schedule_type = "linear"
        self.experiment.rollout.beta.beta_start = 1.0
        self.experiment.rollout.beta.beta_end = 0.0

        # Dagger specific parameters
        self.algo.num_valid_batches = 50
        self.algo.num_eval_games = 100
        self.algo.collect_expert_dataset = False
        self.algo.num_warmup_iterations = 0
        self.algo.num_batches = -1

        # Optimizer parameters
        self.algo.optim_params.policy.optimizer_type = "adam"
        self.algo.optim_params.policy.learning_rate.initial = 1e-4
        self.algo.optim_params.policy.learning_rate.decay_factor = 0.1
        self.algo.optim_params.policy.learning_rate.epoch_schedule = []
        self.algo.optim_params.policy.learning_rate.scheduler_type = "linear"

        # Dataset parameters
        self.train.hdf5_cache_mode = "all"

    def train_config(self):
        """
        Dagger algorithms don't need "next_obs" from hdf5 - so save on storage and compute by disabling it.
        """
        super(DaggerConfig, self).train_config()
        self.train.hdf5_load_next_obs = False
