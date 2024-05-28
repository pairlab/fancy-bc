import torch.nn as nn
import gym
import hydra
import torch
import myosuite.envs.myo.myobase
from hydra.utils import instantiate
from skrl.models.torch import Model, DeterministicMixin, GaussianMixin
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.envs.wrappers.torch import wrap_env
from skrl.trainers.torch import SequentialTrainer


class Policy(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False, 
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum",
                 model_cfg=None):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        if model_cfg is None:
            actor_hidden_dim = 256
            actor_activation_fn = nn.ReLU()
        else:
            actor_hidden_dim = model_cfg["actor_hidden_dim"]
            actor_activation_fn = instantiate(model_cfg["actor_activation_fn"])

        self.net = nn.Sequential(nn.Linear(self.num_observations, actor_hidden_dim),
                                 actor_activation_fn,
                                 nn.Linear(actor_hidden_dim, actor_hidden_dim),
                                 actor_activation_fn)

        self.mean_layer = nn.Linear(actor_hidden_dim, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        self.value_layer = nn.Linear(actor_hidden_dim, 1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        features = self.net(inputs["states"])
        if role == "policy":
            return self.mean_layer(features), self.log_std_parameter, {}
        elif role == "value":
            return self.value_layer(features), {}



@hydra.main(config_path="", config_name="config_skrl",version_base="1.3")
def main(cfg):
    env = gym.vector.make(cfg.env.env_name, num_envs=cfg.num_envs, asynchronous=False)

    env = wrap_env(env, wrapper="gym")

    policy = Policy(env.observation_space, env.action_space, env.device, model_cfg=cfg.model)
    models = {"policy": policy, "value": policy}
    memory = RandomMemory(memory_size=cfg.memory.memory_size, num_envs=env.num_envs, device=env.device)

    agent = PPO(models=models,  # models dict
                memory=memory,  # memory instance, or None if not required
                cfg=cfg.agent,  # configuration dict (preprocessors, learning rate schedulers, etc.)
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=env.device)

    # configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": 40000, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    # start training
    trainer.train()


if __name__ == "__main__":
    main()

