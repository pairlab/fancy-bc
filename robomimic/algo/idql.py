"""
Implementation of Diffusion Policy https://diffusion-policy.cs.columbia.edu/ by Cheng Chi
"""
from typing import Callable, Union, Optional, Tuple
import math
from collections import OrderedDict, deque
from packaging.version import parse as parse_version

import torch
import torch.nn as nn
import torch.nn.functional as F
# requires diffusers==0.11.1
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import einops
# from diffusers.training_utils import EMAModel

import robomimic.models.obs_nets as ObsNets
import robomimic.models.value_nets as ValueNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.ema_model import EMAModel

from robomimic.algo import register_algo_factory_func, PolicyAlgo, ValueAlgo
from robomimic.algo.diffusion_policy import replace_submodules, replace_bn_with_gn, \
    SinusoidalPosEmb, ConditionalResidualBlock1D, Downsample1d, DiffusionPolicyUNet, \
    ConditionalUnet1D

@register_algo_factory_func("idql")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    if algo_config.unet.enabled:
        return IDQLUNet, {}
    elif algo_config.transformer.enabled:
        raise NotImplementedError()
    else:
        raise RuntimeError()

class IDQLUNet(PolicyAlgo, ValueAlgo):
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        super(IDQLUNet, self).__init__()
        # set up different observation groups for @MIMO_MLP
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)
        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)
        
        obs_encoder = ObsNets.ObservationGroupEncoder(
            observation_group_shapes=observation_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )
        # IMPORTANT!
        # replace all BatchNorm with GroupNorm to work with EMA
        # performance will tank if you forget to do this!
        obs_encoder = replace_bn_with_gn(obs_encoder)
        
        obs_dim = obs_encoder.output_shape()[0]

        # create network object
        noise_pred_net = ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=obs_dim*self.algo_config.horizon.observation_horizon,
            diffusion_step_embed_dim=self.algo_config.unet.diffusion_step_embed_dim,
            down_dims=self.algo_config.unet.down_dims,
            kernel_size=self.algo_config.unet.kernel_size,
            n_groups=self.algo_config.unet.n_groups
        )

        critic = nn.ModuleList()
        critic_target = nn.ModuleList()
        for _ in range(self.algo_config.critic.ensemble.n):
            for net_list in (critic, critic_target):
                obs_encoder = ObsNets.ObservationGroupEncoder(
                    observation_group_shapes=observation_group_shapes,
                    encoder_kwargs=encoder_kwargs,
                )
                critic = Conditional1DConvCritic(
                    input_dim=self.ac_dim,
                    global_cond_dim=obs_dim*self.algo_config.horizon.observation_horizon
                )
                net_list.append(nn.ModuleDict({
                    'obs_encoder': obs_encoder,
                    'critic': critic,
                }))

        value = ValueNets.ValueNetwork(
            obs_shapes=self.obs_shapes,
            mlp_layer_dims=self.algo_config.critic.layer_dims,
            goal_shapes=self.goal_shapes,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )

        # the final arch has 2 parts
        nets = nn.ModuleDict({
            'policy': nn.ModuleDict({
                'obs_encoder': obs_encoder,
                'noise_pred_net': noise_pred_net
            }), 
            'critic': critic,
            'critic_target': critic_target,
            'value': value
        })

        nets = nets.float().to(self.device)
        
        # setup noise scheduler
        noise_scheduler = None
        if self.algo_config.ddpm.enabled:
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.algo_config.ddpm.num_train_timesteps,
                beta_schedule=self.algo_config.ddpm.beta_schedule,
                clip_sample=self.algo_config.ddpm.clip_sample,
                prediction_type=self.algo_config.ddpm.prediction_type
            )
        elif self.algo_config.ddim.enabled:
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.algo_config.ddim.num_train_timesteps,
                beta_schedule=self.algo_config.ddim.beta_schedule,
                clip_sample=self.algo_config.ddim.clip_sample,
                set_alpha_to_one=self.algo_config.ddim.set_alpha_to_one,
                steps_offset=self.algo_config.ddim.steps_offset,
                prediction_type=self.algo_config.ddim.prediction_type
            )
        else:
            raise RuntimeError()
        
        # setup EMA
        ema = None
        if self.algo_config.ema.enabled:
            ema = EMAModel(model=nets, power=self.algo_config.ema.power)
                
        # set attrs
        self.nets = nets
        self.noise_scheduler = noise_scheduler
        self.ema = ema
        self.action_check_done = False
        self.obs_queue = None
        self.action_queue = None
    
    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training 
        """
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon

        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, :To, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"][:, :Tp, :]
        
        # check if actions are normalized to [-1,1]
        if not self.action_check_done:
            actions = input_batch["actions"]
            in_range = (-1 <= actions) & (actions <= 1)
            all_in_range = torch.all(in_range).item()
            if not all_in_range:
                raise ValueError('"actions" must be in range [-1,1] for Diffusion Policy! Check if hdf5_normalize_action is enabled.')
            self.action_check_done = True
        
        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)
        
    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        B = batch['actions'].shape[0]
        
        
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(IDQLUNet, self).train_on_batch(batch, epoch, validate=validate)
            actions = batch['actions']
            
            # encode obs
            inputs = {
                'obs': batch["obs"],
                'goal': batch["goal_obs"]
            }
            for k in self.obs_shapes:
                # first two dimensions should be [B, T] for inputs
                assert inputs['obs'][k].ndim - 2 == len(self.obs_shapes[k])
            
            obs_features = TensorUtils.time_distributed(inputs, self.nets['policy']['obs_encoder'], inputs_as_kwargs=True)
            assert obs_features.ndim == 3  # [B, T, D]

            obs_cond = obs_features.flatten(start_dim=1)
            
            # sample noise to add to actions
            noise = torch.randn(actions.shape, device=self.device)
            
            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (B,), device=self.device
            ).long()
            
            # add noise to the clean actions according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = self.noise_scheduler.add_noise(
                actions, noise, timesteps)
            
            # predict the noise residual
            noise_pred = self.nets['policy']['noise_pred_net'](
                noisy_actions, timesteps, global_cond=obs_cond)
            
            # L2 loss
            loss = F.mse_loss(noise_pred, noise)
            
            # logging
            losses = {
                'l2_loss': loss
            }
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                # gradient step
                policy_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets,
                    optim=self.optimizers["policy"],
                    loss=loss,
                )
                
                # update Exponential Moving Average of the model weights
                if self.ema is not None:
                    self.ema.step(self.nets)
                
                step_info = {
                    'policy_grad_norms': policy_grad_norms
                }
                info.update(step_info)

        return info
    
    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(DiffusionPolicyUNet, self).log_info(info)
        log["Loss"] = info["losses"]["l2_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log
    
    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        # setup inference queues
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        obs_queue = deque(maxlen=To)
        action_queue = deque(maxlen=Ta)
        self.obs_queue = obs_queue
        self.action_queue = action_queue
    
    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation [1, Do]
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor [1, Da]
        """
        # obs_dict: key: [1,D]
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon

        # TODO: obs_queue already handled by frame_stack
        # make sure we have at least To observations in obs_queue
        # if not enough, repeat
        # if already full, append one to the obs_queue
        # n_repeats = max(To - len(self.obs_queue), 1)
        # self.obs_queue.extend([obs_dict] * n_repeats)
        
        if len(self.action_queue) == 0:
            # no actions left, run inference
            # turn obs_queue into dict of tensors (concat at T dim)
            # import pdb; pdb.set_trace()
            # obs_dict_list = TensorUtils.list_of_flat_dict_to_dict_of_list(list(self.obs_queue))
            # obs_dict_tensor = dict((k, torch.cat(v, dim=0).unsqueeze(0)) for k,v in obs_dict_list.items())
            
            # run inference
            # [1,T,Da]
            action_sequence = self._get_action_trajectory(obs_dict=obs_dict)
            
            # put actions into the queue
            self.action_queue.extend(action_sequence[0])
        
        # has action, execute from left to right
        # [Da]
        action = self.action_queue.popleft()
        
        # [1,Da]
        action = action.unsqueeze(0)
        return action
        
    def _get_action_trajectory(self, obs_dict, goal_dict=None):
        assert not self.nets.training
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        if self.algo_config.ddpm.enabled is True:
            num_inference_timesteps = self.algo_config.ddpm.num_inference_timesteps
        elif self.algo_config.ddim.enabled is True:
            num_inference_timesteps = self.algo_config.ddim.num_inference_timesteps
        else:
            raise ValueError
        
        # select network
        nets = self.nets
        if self.ema is not None:
            nets = self.ema.averaged_model
        
        # encode obs
        inputs = {
            'obs': obs_dict,
            'goal': goal_dict
        }
        for k in self.obs_shapes:
            # first two dimensions should be [B, T] for inputs
            assert inputs['obs'][k].ndim - 2 == len(self.obs_shapes[k])
        obs_features = TensorUtils.time_distributed(inputs, self.nets['policy']['obs_encoder'], inputs_as_kwargs=True)
        assert obs_features.ndim == 3  # [B, T, D]
        B = obs_features.shape[0]

        # reshape observation to (B,obs_horizon*obs_dim)
        obs_cond = obs_features.flatten(start_dim=1)

        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (B, Tp, action_dim), device=self.device)
        naction = noisy_action
        
        # init scheduler
        self.noise_scheduler.set_timesteps(num_inference_timesteps)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            noise_pred = nets['policy']['noise_pred_net'](
                sample=naction, 
                timestep=k,
                global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample
        
        # process action using Ta
        start = To - 1
        end = start + Ta
        action = naction[:,start:end]
        return action

    def serialize(self):
        """
        Get dictionary of current model parameters.
        """
        return {
            "nets": self.nets.state_dict(),
            "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
        }

    def deserialize(self, model_dict):
        """
        Load model from a checkpoint.

        Args:
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
        """
        self.nets.load_state_dict(model_dict["nets"])
        if model_dict.get("ema", None) is not None:
            self.ema.averaged_model.load_state_dict(model_dict["ema"])
        

# =================== Vision Encoder Utils =====================

# =================== UNet for Diffusion ==============

# class Conditional1DConvDoubleCritic(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         self.q1_model = Conditional1DConvCritic(*args, **kwargs)
#         self.q2_model = Conditional1DConvCritic(*args, **kwargs)

#     def forward(self, *args, **kwargs):
#         return self.q1_model(*args, **kwargs), self.q2_model(*args, **kwargs)
    
#     def q1(self, *args, **kwargs):
#         return self.q1_model(*args, **kwargs)

#     def q2(self, *args, **kwargs):
#         return self.q1_model(*args, **kwargs)

#     def q_min(self, *args, **kwargs):
        q1, q2 = self.forward(*args, **kwargs)
        return torch.min(q1, q2)

class Conditional1DConvCritic(nn.Module):
    def __init__(self, 
        input_dim,
        global_cond_dim,
        down_dims=[256,512,1024],
        mlp_hidden_dim=256,
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=False
        ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)

        cond_dim = global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        self.mlp = nn.Sequential(
            nn.Linear(down_dims[-1] + cond_dim, mlp_hidden_dim),
            nn.Mish(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.Mish(),
            nn.Linear(mlp_hidden_dim, 1))

        self.down_modules = down_modules

        print("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def forward(self, 
            actions: torch.Tensor, 
            global_cond=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        actions = einops.rearrange(actions, 'b h t -> b t h')

        x = actions
        for _, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_cond)
            x = resnet2(x, global_cond)
            x = downsample(x)

        x = torch.max(x, dim=2)[0]
        x = torch.cat([x, global_cond], dim=-1)
        out = self.mlp(x)
        
        return out

