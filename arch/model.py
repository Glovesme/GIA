import numpy as np
import torch
from torch import nn
import math
import torchvision
from collections import OrderedDict

from arch.VariationalBottleneck import VariationalBottleneck
from arch.BayesianLayer import BayesLinear


# class VBMLP(nn.Module):
#     def __init__(self, width=512):
#         super().__init__()
#         self.width = width
#         self.num_classes = 10
#         self.data_shape = (3, 32, 32)
#
#         self.flat = nn.Flatten()
#         self.in_linear = nn.Linear(math.prod(self.data_shape), width)
#         self.relu = nn.ReLU()
#         self.out_linear = nn.Linear(width, self.num_classes)
#
#         self.is_freeze = False
#
#         self.vb = VariationalBottleneck(in_shape=(width,))
#         self.learned_eps = torch.nn.Parameter(torch.randn(size=(1, 256)))
#
#     def forward(self, x):
#         x = self.flat(x)
#         x = self.in_linear(x)
#         x = self.relu(x)
#         x = self.vb(x, self.learned_eps if self.is_freeze else None)
#         x = self.out_linear(x)
#         return x
#
#     def freeze(self):
#         self.is_freeze = True
#
#     def unfreeze(self):
#         self.is_freeze = False
#
#     def loss(self):
#         return self.vb.loss()
#
#     def clear(self):
#         self.eps = torch.randn(size=(1, 256))


class VBMLP(nn.Module):
    def __init__(self, width=1024):
        super().__init__()
        self.width = width
        self.num_classes = 10
        self.data_shape = (3, 32, 32)

        self.flat = nn.Flatten()
        self.l1 = nn.Linear(np.prod(self.data_shape), width)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(width, width)
        self.l3 = nn.Linear(width, self.num_classes)

        self.is_freeze = False

        self.vb = VariationalBottleneck(in_shape=(width,))
        self.learned_eps = torch.nn.Parameter(torch.randn(size=(1, 256)))

    def forward(self, x):
        x = self.flat(x)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.vb(x, self.learned_eps if self.is_freeze else None)
        x = self.l3(x)
        return x

    def freeze(self):
        self.is_freeze = True

    def unfreeze(self):
        self.is_freeze = False

    def loss(self):
        return self.vb.loss()

    def clear(self):
        self.eps = torch.randn(size=(1, 256))