#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
*************************************************
@Project -> File   ：pytorch_learn -> nn_lose
@IDE    ：PyCharm
@Author ：Zhuge hammer
@Date   ：2024/1/23 15:59
@Desc   ：
*************************************************
"""
import torch
from torch.nn import L1Loss, MSELoss

input = torch.tensor([[1, 2, 3]], dtype=torch.float32)
target = torch.tensor([[1.5, 2.5, 3.5]], dtype=torch.float32)

input = torch.reshape(input, (-1, 1, 1, 3))
target = torch.reshape(target, (-1, 1, 1, 3))

loss_l1 = L1Loss()
res_l1 = loss_l1(input, target)
print(res_l1)

loss_mse = MSELoss()
res_mse = loss_mse(input, target)
print(res_mse)