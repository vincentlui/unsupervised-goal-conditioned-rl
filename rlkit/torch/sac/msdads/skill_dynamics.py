import torch
from torch import nn as nn
from torch.nn import functional as F, BatchNorm1d
from torch.distributions import Normal, Independent, Categorical

from rlkit.torch.networks import Mlp
import numpy as np
from rlkit.torch.sac.gcs.networks import BNMlp, MixtureSameFamily

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


def identity(x):
    return x


class SkillDynamics(BNMlp):
    def __init__(
            self,
            hidden_sizes,
            input_size,
            state_dim,
            num_components=1,
            std=None,
            init_w=1e-3,
            batch_norm=False,
            **kwargs
    ):
        output_size = state_dim * num_components
        super().__init__(
            hidden_sizes=hidden_sizes,
            input_size=input_size,
            output_size=output_size,
            init_w=init_w,
            batch_norm=batch_norm,
            **kwargs
        )
        # self.skill_dim = skill_dim
        self.state_dim = state_dim
        self.log_std = None
        self.std = std
        self.batch_norm = batch_norm
        self.batchnorm_output = BatchNorm1d(state_dim, affine=False)
        self.num_components = num_components

        if std is None:
            last_hidden_size = input_size
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, output_size)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.std = torch.Tensor(np.array(std))
            self.log_std = torch.log(self.std)
            assert np.all(LOG_SIG_MIN <= self.log_std) and np.all(self.log_std <= LOG_SIG_MAX)

        if num_components > 1:
            self.last_fc_categorical = nn.Linear(hidden_sizes[-1], num_components)
            self.last_fc_categorical.weight.data.uniform_(-init_w, init_w)
            self.last_fc_categorical.bias.data.uniform_(-init_w, init_w)

    def forward(
            self, input, return_preactivations=False,
    ):
        h = input
        if self.batch_norm:
            h = self.batchnorm_input(h)
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
            if self.batch_norm:
                h = self.batchnorm_hidden[i](h)
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std

        if self.num_components > 1:
            mean = mean.view(-1, self.num_components, self.state_dim)
            std = std.view(-1, self.num_components, self.state_dim)
            categorical_logits = self.last_fc_categorical(h)
            distribution = MixtureSameFamily(
                Categorical(logits=categorical_logits),
                Independent(Normal(mean, std), 1)
            )
        else:
            distribution = Independent(Normal(mean, std), 1)

        return distribution

    def log_prob(self, input, target):
        sf_distribution = self(input)
        if self.batch_norm:
            target = self.batchnorm_output(target)
        log_likelihood = sf_distribution.log_prob(target)
        return log_likelihood
