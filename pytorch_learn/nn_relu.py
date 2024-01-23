#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
*************************************************
@Project -> File   ：pytorch_learn -> nn_relu
@IDE    ：PyCharm
@Author ：Zhuge hammer
@Date   ：2024/1/23 14:21
@Desc   ：
*************************************************
"""
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                      [-1, 3]])

input = torch.reshape(input, (-1, 1, 2, 2))

dataset = torchvision.datasets.CIFAR10(root='dataset', train=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


class hammer(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # out = self.relu(x)
        out = self.sigmoid(x)
        return out


HA = hammer()
writer = SummaryWriter('./logs')

step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images('sigmoid_In', imgs, step)
    outputs = HA(imgs)
    writer.add_images('sigmoid_Out', outputs, step)
    step += 1

writer.close()

