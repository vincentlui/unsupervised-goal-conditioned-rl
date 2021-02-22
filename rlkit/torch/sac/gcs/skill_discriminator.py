import torch
from torch import nn as nn
from torch.nn import functional as F

from rlkit.policies.base import Policy
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import eval_np
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.modules import LayerNorm
from rlkit.torch.networks import Mlp
import numpy as np

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


def identity(x):
    return x


class SkillDiscriminator(Mlp):
    def __init__(
            self,
            hidden_sizes,
            input_size,
            skill_dim,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        super().__init__(
            hidden_sizes=hidden_sizes,
            input_size=input_size,
            output_size=skill_dim,
            init_w=init_w,
            **kwargs
        )
        self.skill_dim = skill_dim
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = input_size
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, skill_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.std = torch.Tensor(np.array(std))
            self.log_std = torch.log(self.std)
            assert np.all(LOG_SIG_MIN <= self.log_std) and np.all(self.log_std <= LOG_SIG_MAX)

    def forward(
            self,
            input,
            return_log_prob=False,
    ):
        h = input
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std

        return mean, std