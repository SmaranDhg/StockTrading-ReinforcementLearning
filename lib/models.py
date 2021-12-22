# %%
"""---------------------------Dueling Architectecture---------------------------"""
# Q(s,a)=V(s)+A(s,a) - A(s,a).mean()
# last part effectively enforeces teh A(s,a).mean() to be zero
# i.e average value of the advantage per of state action to be zero
from torch import nn
import torch

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNFF(nn.Module):
    def __init__(self, obs_len, actions_n):

        super().__init__()

        # dueling architecture
        self.fc_val = nn.Sequential(
            nn.Linear(obs_len, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.fc_adv = nn.Sequential(
            nn.Linear(obs_len, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, actions_n),
        )

    def forward(self, x):
        val = self.fc_val(x)
        adv = self.fc_adv(x)
        return val + adv - adv.mean()


class DQNC1D(nn.Module):
    def __init__(self, shape, actions_n):
        super().__init__()

        # feature atraction layer with actions of 1d convolution

        self.conv = nn.Sequential(
            nn.Conv1d(shape[0], 128, 5),
            nn.Relu(),
            nn.Conv1d(125, 145, 5),
            nn.Relu(),
        )

        out_size = self._get_conv_out(shape)

        self.fc_val = nn.Sequential(
            nn.Linear(out_size, 512), nn.ReLU(), nn.Linear(512, 1)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(out_size, 512), nn.ReLU(), nn.Linear(512, actions_n)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size))

    def forward(self, x):
        # -- todo
        conv_out = self.conv(x).view(x.size()[0], -1)  # batch,out
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + adv - adv.mean()
