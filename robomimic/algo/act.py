"""
Implementation of Action Chunking with Transformers (ACT).
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import robomimic.utils.tensor_utils as TensorUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo
from robomimic.algo.bc import BC_VAE


@register_algo_factory_func("act")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    algo_class, algo_kwargs = ACT, {}

    return algo_class, algo_kwargs


class ACT(BC_VAE):
    """
    BC training with a VAE policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.nets = nn.ModuleDict()
        torch.set_default_dtype(torch.float32)
        self.chunk_size = self.global_config["train"]["seq_length"]
        self.camera_keys = self.obs_config['modalities']['obs']['rgb'].copy()
        self.proprio_keys = self.obs_config['modalities']['obs']['low_dim'].copy()
        self.obs_keys = self.proprio_keys + self.camera_keys
        self.action_key = self.global_config.train.action_keys[0]

        self.proprio_dim = 0
        self.use_vq = self.algo_config.act.vq
        for k in self.proprio_keys:
            self.proprio_dim += self.obs_key_shapes[k][0]

        from detr.main import build_ACT_model_and_optimizer
        from detr.models.latent_model import Latent_Model_Transformer

        policy_config = {'num_queries': self.chunk_size,
                         'hidden_dim': self.algo_config.act.hidden_dim,
                         'dim_feedforward': self.algo_config.act.dim_feedforward,
                         'backbone': self.algo_config.act.backbone,
                         'enc_layers': self.algo_config.act.enc_layers,
                         'dec_layers': self.algo_config.act.dec_layers,
                         'vq': self.algo_config.act.vq,
                         'vq_dim': self.algo_config.act.vq_dim,
                         'vq_class': self.algo_config.act.vq_class,
                         'nheads': self.algo_config.act.nheads,
                         'latent_dim': self.algo_config.act.latent_dim,
                         'a_dim': self.ac_dim,
                         'state_dim': self.proprio_dim,
                         'camera_names': self.camera_keys,
                         'ckpt_dir': self.global_config.train.output_dir,
                         'task_name': self.global_config.experiment.name,
                         'seed': self.global_config.train.seed,
                         'num_steps': self.global_config.train.num_epochs,
                         'action_dim': self.global_config.train.action_config[self.action_key]['dim']
                         }

        self.kl_weight = self.algo_config.loss.kl_weight
        self.vq_l1_weight = self.algo_config.loss.vq_l1_weight
        self.vq_weight = self.algo_config.loss.vq_weight
        self.vq_ce_weight = self.algo_config.loss.vq_ce_weight
        model, optimizer = build_ACT_model_and_optimizer(policy_config)
        self.nets["policy"] = model
        if self.use_vq:
            self.nets["latent_model"] = Latent_Model_Transformer(policy_config['vq_dim'], policy_config['vq_dim'], policy_config['vq_class'])
        self.nets = self.nets.float().to(self.device)

        self.temporal_agg = False
        self.query_frequency = self.chunk_size  # TODO maybe tune

        self._step_counter = 0
        self.a_hat_store = None

    def deserialize(self, model_dict):
        """
        Load model from a checkpoint.

        Args:
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
        """
        self.nets.load_state_dict(model_dict, strict=False)

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

        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, 0, :] for k in batch["obs"] if k != 'pad_mask'}
        input_batch["obs"]['pad_mask'] = batch["obs"]['pad_mask']
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch[self.action_key] = batch[self.action_key][:, :, :]
        # we move to device first before float conversion because image observation modalities will be uint8 -
        # this minimizes the amount of data transferred to GPU
        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Update from superclass to set categorical temperature, for categorcal VAEs.
        """

        return super(BC_VAE, self).train_on_batch(batch, epoch, validate=validate)

    @torch.no_grad()
    def vq_encode(self, qpos, actions, is_pad):
        actions = actions[:, :self.nets["policy"].num_queries]
        is_pad = is_pad[:, :self.nets["policy"].num_queries]

        _, _, binaries, _, _ = self.nets["policy"].encode(qpos, actions, is_pad)

        return binaries

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            predictions (dict): dictionary containing network outputs
        """

        proprio = [batch["obs"][k] for k in self.proprio_keys]
        proprio = torch.cat(proprio, axis=1)
        qpos = proprio

        images = []
        for cam_name in self.camera_keys:
            image = batch['obs'][cam_name]
            image = self.normalize(image)
            image = image.unsqueeze(axis=1)
            images.append(image)
        images = torch.cat(images, axis=1)

        env_state = torch.zeros([qpos.shape[0], 10]).cuda()  # this is not used

        actions = batch[self.action_key]
        is_pad = batch['obs']['pad_mask'] == 0  # from 1.0 or 0 to False and True
        is_pad = is_pad.squeeze(dim=-1)

        a_hat, is_pad_hat, (mu, logvar), probs, gt_labels = self.nets["policy"](qpos, images, env_state, actions, is_pad)
        loss_dict = dict()
        if self.use_vq:
            total_kld = [torch.tensor(0.0)]
        else:
            total_kld, dim_wise_kld, mean_kld = self.kl_divergence(mu, logvar)
        loss_dict['kl'] = total_kld[0]

        all_l1 = F.l1_loss(actions, a_hat, reduction='none')
        l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
        loss_dict['l1'] = l1

        predictions = OrderedDict(
            actions=actions,
            kl_loss=loss_dict['kl'],
            reconstruction_loss=loss_dict['l1'],
        )
        if self.use_vq:
            predictions["probs"] = probs
            predictions["gt_labels"] = gt_labels.detach()

        return predictions

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

        proprio = [obs_dict[k] for k in self.proprio_keys]
        proprio = torch.cat(proprio, axis=1)
        qpos = proprio

        images = []
        for cam_name in self.camera_keys:
            image = obs_dict[cam_name]
            image = self.normalize(image)
            image = image.unsqueeze(axis=1)
            images.append(image)
        images = torch.cat(images, axis=1)

        env_state = torch.zeros([qpos.shape[0], 10]).cuda() # not used
        if self.use_vq:
            vq_sample = self.nets["latent_model"].generate(1, temperature=1, x=None)
        else:
            vq_sample = torch.zeros([qpos.shape[0], self.nets["policy"].vq_class * self.nets["policy"].vq_dim]).cuda()

        if self._step_counter % self.query_frequency == 0:
            a_hat, is_pad_hat, (mu, logvar), probs, gt_labels = self.nets["policy"](qpos, images, env_state, vq_sample=vq_sample)
            self.a_hat_store = a_hat

        action = self.a_hat_store[:, self._step_counter % self.query_frequency, :]
        self._step_counter += 1
        return action


    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        self._step_counter = 0

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.
        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # total loss is sum of reconstruction and KL, weighted by beta
        kl_loss = predictions["kl_loss"]
        recons_loss = predictions["reconstruction_loss"]
        action_loss = recons_loss + self.kl_weight * kl_loss
        losses = OrderedDict(
            recons_loss=recons_loss,
            kl_loss=kl_loss,
        )
        if self.use_vq:
            probs = predictions["probs"]
            gt_labels = predictions["gt_labels"]
            vq_l1, vq_ce = self.compute_vq_loss(probs, gt_labels) 
            vq_discrepancy = vq_l1 * self.vq_l1_weight + vq_ce * self.vq_ce_weight
            action_loss = action_loss + vq_discrepancy * self.vq_weight
            losses["vq_discrepancy"] = vq_discrepancy
        losses["action_loss"] = action_loss
        return losses

    def compute_vq_loss(self, probs, gt_labels):
        # F.l1_loss(probs, gt_labels, reduction='mean')
        if self.vq_l1_weight > 0:
            l1_loss = F.l1_loss(probs, gt_labels, reduction='mean')
        else:
            l1_loss = 0
        if self.vq_ce_weight > 0:
            ce_loss = F.cross_entropy(probs, gt_labels)
        else:
            ce_loss = 0
        return l1_loss, ce_loss

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.
        Args:
            info (dict): dictionary of info
        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["KL_Loss"] = info["losses"]["kl_loss"].item()
        log["Reconstruction_Loss"] = info["losses"]["recons_loss"].item()
        if self.use_vq:
            log["VQ_Loss"] = info["losses"]["vq_discrepancy"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    def kl_divergence(self, mu, logvar):
        batch_size = mu.size(0)
        assert batch_size != 0
        if mu.data.ndimension() == 4:
            mu = mu.view(mu.size(0), mu.size(1))
        if logvar.data.ndimension() == 4:
            logvar = logvar.view(logvar.size(0), logvar.size(1))

        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        total_kld = klds.sum(1).mean(0, True)
        dimension_wise_kld = klds.mean(0)
        mean_kld = klds.mean(1).mean(0, True)

        return total_kld, dimension_wise_kld, mean_kld

