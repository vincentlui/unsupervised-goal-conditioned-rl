import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from rlkit.policies.base import ExplorationPolicy, Policy
from rlkit.torch.core import eval_np
from rlkit.torch.distributions import TanhNormal
from rlkit.torch.networks import Mlp

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class DiscretePolicy(Mlp, ExplorationPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            init_w=1e-3,
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.output_activation = F.softmax

    def get_action(self, obs_np, deterministic=False, return_log_prob=True):
        if return_log_prob and not deterministic:
            actions, log_probs = self.get_actions(obs_np[None], deterministic=deterministic, return_log_prob=return_log_prob)
            return actions[0, :], {"log_prob": log_probs[0, :]}
        else:
            actions = self.get_actions(obs_np[None], deterministic=deterministic, return_log_prob=return_log_prob)
            return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False, return_log_prob=True):
        outputs = eval_np(self, obs_np, deterministic=deterministic, return_log_prob=return_log_prob)
        if return_log_prob and not deterministic:
            return outputs[0], outputs[1]
        else:
            return outputs[0]

    def forward(
            self,
            obs,
            reparameterize=False,
            deterministic=False,
            return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        softmax_probs = self.output_activation(self.last_fc(h))

        log_prob = None
        if deterministic:
            action = torch.zeros(obs.shape[0], self.output_size)
            action[:, torch.argmax(softmax_probs)] = 1
        else:
            action = torch.zeros(obs.shape[0], self.output_size)
            categorical =  Categorical(softmax_probs)
            index = categorical.sample()
            if return_log_prob:
                log_prob = torch.zeros(obs.shape[0], 1)
                log_prob[:, :] = categorical.log_prob(index)
            action[:, index] = 1
        return (
            action, log_prob, softmax_probs
        )

class TanhGaussianPolicy(Mlp, ExplorationPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs_np, deterministic=False, return_log_prob=True):
        obs_np = np.array(obs_np)
        if return_log_prob and not deterministic:
            actions, log_probs = self.get_actions(obs_np[None], deterministic=deterministic, return_log_prob=return_log_prob)
            return actions[0, :], {"log_prob": log_probs[0, :]}
        else:
            actions = self.get_actions(obs_np[None], deterministic=deterministic, return_log_prob=return_log_prob)
            return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False, return_log_prob=True):
        outputs = eval_np(self, obs_np, deterministic=deterministic, return_log_prob=return_log_prob)
        if return_log_prob and not deterministic:
            return outputs[0], outputs[3]
        else:
            return outputs[0]

    def forward(
            self,
            obs,
            reparameterize=False,
            deterministic=False,
            return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = obs
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
                action, pre_tanh_value = tanh_normal.sample(
                    return_pretanh_value=True
                )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                action = tanh_normal.sample()
        return (
            action, mean, log_std, log_prob, entropy, std,
            mean_action_log_prob, pre_tanh_value,
        )

class MakeDeterministic(Policy):
    def __init__(self, stochastic_policy):
        self.stochastic_policy = stochastic_policy

    def get_action(self, observation):
        return self.stochastic_policy.get_action(observation,
                                                 deterministic=True)
