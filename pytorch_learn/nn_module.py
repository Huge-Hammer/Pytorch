#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
*************************************************
@Project -> File   ：pytorch_learn -> nn_learn
@IDE    ：PyCharm
@Author ：Zhuge hammer
@Date   ：2024/1/23 9:48
@Desc   ：
*************************************************
"""
import torch
from torch import nn

class hammer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = x + 1
        return out

if __name__ == '__main__':
    HA = hammer()
    x = torch.tensor(1.)
    out = HA(x)
    print(out)



