#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
*************************************************
@Project -> File   ：pytorch_learn -> nn_conv2d
@IDE    ：PyCharm
@Author ：Zhuge hammer
@Date   ：2024/1/23 10:30
@Desc   ：
*************************************************
"""
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10('./dataset', train=False, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)


class hammer(nn.Module):
    def __init__(self):
        super(hammer, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        return x

HA = hammer()
writer = SummaryWriter('./logs')

step = 0
for data in dataloader:
    imgs, targets = data
    outputs = HA(imgs)
    # img, torch.Size([64, 3, 32, 32])
    writer.add_images('conv2d_In', imgs, step)
    # outputs, torch.Size([64, 6, 30, 30])
    outputs = torch.reshape(outputs, (-1, 3, 30, 30))
    # outputs, torch.Size([64, 3, 30, 30])
    writer.add_images('conv2d_Out', outputs, step)
    step += 1

writer.close()