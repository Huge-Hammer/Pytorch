#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
*************************************************
@Project -> File   ：pytorch_learn -> nn.maxpool
@IDE    ：PyCharm
@Author ：Zhuge hammer
@Date   ：2024/1/23 11:26
@Desc   ：
*************************************************
"""
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype=torch.float32)
#
# input = torch.reshape(input, (-1, 1, 5, 5))


class hammer(nn.Module):
    def __init__(self):
        super(hammer, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, x):
        x = self.maxpool(x)
        return x


dataset = torchvision.datasets.CIFAR10('./dataset', train=False, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset=dataset, batch_size=64)

HA = hammer()

writer = SummaryWriter('./logs')

step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images('maxpool_In', imgs, step)
    outputs = HA(imgs)
    writer.add_images('maxpool_Out', outputs, step)
    step += 1

writer.close()
