#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
*************************************************
@Project -> File   ：pytorch_learn -> nn_conv
@IDE    ：PyCharm
@Author ：Zhuge hammer
@Date   ：2024/1/23 10:05
@Desc   ：
*************************************************
"""
import torch
import torch.nn.functional as F

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

input = input.reshape((1, 1, 5, 5))
kernel = kernel.reshape((1, 1, 3, 3))

out = F.conv2d(input, kernel, stride=1)
print(out)

out2 = F.conv2d(input, kernel, stride=2)
print(out2)

out3 = F.conv2d(input, kernel, stride=1, padding=1)
print(out3)

out4 = F.conv2d(input, kernel, stride=1, padding=(1, 2))
print(out4)

out5 = F.conv2d(input, kernel, stride=1, padding=(2, 1))
print(out5)