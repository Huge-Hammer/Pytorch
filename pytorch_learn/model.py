#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
*************************************************
@Project -> File   ：pytorch_learn -> model
@IDE    ：PyCharm
@Author ：Zhuge hammer
@Date   ：2024/1/23 19:49
@Desc   ：
*************************************************
"""
import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear


class hammer(nn.Module):
    def __init__(self):
        super(hammer, self).__init__()
        self.model = Sequential(
            Conv2d(3, 32, 5, 1, 2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, 1, 2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, 1, 2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    HA = hammer()
    input = torch.ones((64, 3, 32, 32))
    output = HA(input)
    print(output.shape)