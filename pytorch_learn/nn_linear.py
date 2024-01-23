#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
*************************************************
@Project -> File   ：pytorch_learn -> nn_linear
@IDE    ：PyCharm
@Author ：Zhuge hammer
@Date   ：2024/1/23 14:44
@Desc   ：
*************************************************
"""
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='dataset', train=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


class hammer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(196608, 10)

    def forward(self, x):
        out = self.linear(x)
        return out


HA = hammer()
writer = SummaryWriter('./logs')

step = 0
for data in dataloader:
    imgs, targets = data
    out = torch.flatten(imgs)
    print(out.shape)
