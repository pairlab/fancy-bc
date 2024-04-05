"""
Config for Diffusion Policy algorithm.
"""

from robomimic.config.base_config import BaseConfig
from robomimic.config.config import Config

class DiffusionPolicyConfig(BaseConfig):
    ALGO_NAME = "diffusion_policy"
    
    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the 
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config` 
        argument to the constructor. Any parameter that an algorithm needs to determine its 
        training and test-time behavior should be populated here.
        """
        
        # optimization parameters
        self.algo.optim_params.policy.learning_rate.initial = 1e-4      # policy learning rate
        self.algo.optim_params.policy.learning_rate.decay_factor = 0.1  # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.policy.learning_rate.epoch_schedule = [] # epochs where LR decay occurs
        self.algo.optim_params.policy.regularization.L2 = 0.00          # L2 regularization strength

        # horizon parameters
        self.algo.horizon.observation_horizon = 2
        self.algo.horizon.action_horizon = 8
        self.algo.horizon.prediction_horizon = 16
        
        # UNet parameters
        self.algo.unet.enabled = True
        self.algo.unet.diffusion_step_embed_dim = 256
        self.algo.unet.down_dims = [256,512,1024]
        self.algo.unet.kernel_size = 5
        self.algo.unet.n_groups = 8
        
        # EMA parameters
        self.algo.ema.enabled = True
        self.algo.ema.power = 0.75

        # AMP parameters
        self.algo.amp.enabled = False
        
        # Noise Scheduler
        ## DDPM
        self.algo.ddpm.enabled = True
        self.algo.ddpm.num_train_timesteps = 100
        self.algo.ddpm.num_inference_timesteps = 100
        self.algo.ddpm.beta_schedule = 'squaredcos_cap_v2'
        self.algo.ddpm.clip_sample = True
        self.algo.ddpm.prediction_type = 'epsilon'

        ## DDIM
        self.algo.ddim.enabled = False
        self.algo.ddim.num_train_timesteps = 100
        self.algo.ddim.num_inference_timesteps = 10
        self.algo.ddim.beta_schedule = 'squaredcos_cap_v2'
        self.algo.ddim.clip_sample = True
        self.algo.ddim.set_alpha_to_one = True
        self.algo.ddim.steps_offset = 0
        self.algo.ddim.prediction_type = 'epsilon'

    def train_config(self):
        super().train_config()

        # One of ["all", "low_dim", or None]. Set to "all" to cache entire hdf5 in memory - this is 
        # by far the fastest for data loading. Set to "low_dim" to cache all non-image data. Set
        # to None to use no caching - in this case, every batch sample is retrieved via file i/o.
        # You should almost never set this to None, even for large image datasets.
        self.train.hdf5_cache_mode = "low_dim"

        # whether to load "next_obs" group from hdf5 - only needed for batch / offline RL algorithms
        self.train.hdf5_load_next_obs = False

        # if provided, use the list of demo keys under the hdf5 group "mask/@hdf5_validation_filter_key" for validation.
        # Must be provided if @experiment.validate is True.
        self.train.hdf5_validation_filter_key = None

        # length of experience sequence to fetch from the dataset
        # and whether to pad the beginning / end of the sequence at boundaries of trajectory in dataset
        self.train.seq_length = 15
        self.train.frame_stack = 2

        # keys from hdf5 to load into each batch, besides "obs" and "next_obs". If algorithms
        # require additional keys from each trajectory in the hdf5, they should be specified here.
        self.train.dataset_keys = (
            "actions", 
        )

        ## learning config ##
        self.train.batch_size = 256     # batch size
    
    def observation_config(self):
        super().observation_config()
        """
        This function populates the `config.observation` attribute of the config, and is given 
        to the `Algo` subclass (see `algo/algo.py`) for each algorithm through the `obs_config` 
        argument to the constructor. This portion of the config is used to specify what 
        observation modalities should be used by the networks for training, and how the 
        observation modalities should be encoded by the networks. While this class has a 
        default implementation that usually doesn't need to be overriden, certain algorithm 
        configs may choose to, in order to have seperate configs for different networks 
        in the algorithm. 
        """

        # =============== RGB default encoder (ResNet backbone + linear layer output) ===============
        self.observation.encoder.rgb.core_kwargs.feature_dimension = 64
        self.observation.encoder.rgb.core_kwargs.backbone_class = "ResNet18Conv"
        self.observation.encoder.rgb.core_kwargs.backbone_kwargs = Config()
        self.observation.encoder.rgb.core_kwargs.backbone_kwargs.pretrained = False
        self.observation.encoder.rgb.core_kwargs.backbone_kwargs.input_coord_conv = False
        self.observation.encoder.rgb.core_kwargs.pool_class = "SpatialSoftmax"
        self.observation.encoder.rgb.core_kwargs.pool_kwargs = Config()
        self.observation.encoder.rgb.core_kwargs.pool_kwargs.num_kp = 32
        self.observation.encoder.rgb.core_kwargs.pool_kwargs.learnable_temperature = False
        self.observation.encoder.rgb.core_kwargs.pool_kwargs.temperature = 1.0
        self.observation.encoder.rgb.core_kwargs.pool_kwargs.noise_std = 0.0

        # RGB: Obs Randomizer settings
        self.observation.encoder.rgb.obs_randomizer_class = "CropRandomizer"                # Can set to 'CropRandomizer' to use crop randomization
        self.observation.encoder.rgb.obs_randomizer_kwargs.crop_height = 76
        self.observation.encoder.rgb.obs_randomizer_kwargs.crop_width = 76
        self.observation.encoder.rgb.obs_randomizer_kwargs.num_crops = 1
        self.observation.encoder.rgb.obs_randomizer_kwargs.pos_enc = False
