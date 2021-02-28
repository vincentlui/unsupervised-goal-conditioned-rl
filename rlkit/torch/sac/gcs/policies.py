import numpy as np
import torch
from torch import nn as nn
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.uniform import Uniform
import random

from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.policies.base import Policy
from rlkit.torch.core import eval_np
from rlkit.torch.distributions import TanhNormal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class SkillTanhGaussianPolicy(TanhGaussianPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            skill_dim=10,
            **kwargs
    ):
        super().__init__(
            hidden_sizes=hidden_sizes,
            obs_dim=obs_dim,
            action_dim=action_dim,
            std=std,
            init_w=init_w,
            **kwargs
        )
        self.skill_dim = skill_dim
        self.skill = random.randint(0, self.skill_dim - 1)
        self.skill_vec = np.zeros(self.skill_dim)
        self.skill_vec[self.skill] += 1

    def get_action(self, obs_np, deterministic=False, return_log_prob=False):
        # generate (skill_dim, ) matrix that stacks one-hot skill vectors
        # online reinforcement learning
        obs_np = np.concatenate((obs_np, self.skill_vec), axis=0)
        action, _, _, log_prob, *_ = self.get_actions(obs_np[None], deterministic=deterministic,
                                                      return_log_prob=return_log_prob)
        if return_log_prob:
            return action[0, :], {"skill": self.skill_vec,
                                  "log_prob": log_prob[0]}  # , "pre_tanh_value": pre_tanh_value[0,:]}
        return action[0, :], {"skill": self.skill_vec}

    def get_actions(self, obs_np, deterministic=False, return_log_prob=False):
        return eval_np(self, obs_np, deterministic=deterministic, return_log_prob=return_log_prob)

    def set_skill(self, skill):
        self.skill = skill
        self.skill_vec = np.zeros(self.skill_dim)
        self.skill_vec[self.skill] += 1

    def skill_reset(self):
        self.skill = random.randint(0, self.skill_dim-1)
        self.skill_vec = np.zeros(self.skill_dim)
        self.skill_vec[self.skill] += 1

    def forward(
            self,
            obs,
            skill_vec=None,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        if skill_vec is None:
            h = obs
        else:
            h = torch.cat((obs, skill_vec), dim=1)

        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return (
            action, mean, log_std, log_prob, entropy, std,
            mean_action_log_prob, pre_tanh_value,
        )


class UniformSkillTanhGaussianPolicy(SkillTanhGaussianPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            skill_dim=2,
            low=[-1, -1],
            high=[1, 1],
            **kwargs
    ):
        super().__init__(
            hidden_sizes=hidden_sizes,
            obs_dim=obs_dim,
            action_dim=action_dim,
            std=std,
            init_w=init_w,
            skill_dim=skill_dim,
            **kwargs
        )
        self.skill_space = Uniform(torch.Tensor(low), torch.Tensor(high))
        self.skill = self.skill_space.sample().cpu().numpy()

    # def get_action(self, obs_np, deterministic=False):
    #     # generate (skill_dim, ) matrix that stacks one-hot skill vectors
    #     # online reinforcement learning
    #     obs_np = np.concatenate((obs_np, self.skill), axis=0)
    #     actions = self.get_actions(obs_np[None], deterministic=deterministic)
    #     return actions[0, :], {"skill": self.skill}
    def get_action(self, obs_np, deterministic=False, return_log_prob=False):
        # generate (skill_dim, ) matrix that stacks one-hot skill vectors
        # online reinforcement learning
        obs_np = np.concatenate((obs_np, self.skill), axis=0)
        action, _, _, log_prob, *_ = self.get_actions(obs_np[None], deterministic=deterministic, return_log_prob=return_log_prob)
        if return_log_prob:
            return action[0,:], {"skill": self.skill, "log_prob": log_prob[0]}#, "pre_tanh_value": pre_tanh_value[0,:]}
        return action[0,:], {"skill": self.skill}

    def get_actions(self, obs_np, deterministic=False, return_log_prob=False):
        return eval_np(self, obs_np, deterministic=deterministic, return_log_prob=return_log_prob)

    def skill_reset(self):
        self.skill = self.skill_space.sample().cpu().numpy()

    def reset(self):
        self.skill_reset()
        super(UniformSkillTanhGaussianPolicy, self).reset()


class MakeDeterministic(Policy):
    def __init__(self, stochastic_policy):
        self.stochastic_policy = stochastic_policy

    def get_action(self, observation):
        return self.stochastic_policy.get_action(observation,
                                                 deterministic=True)
