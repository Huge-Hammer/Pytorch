#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
*************************************************
@Project -> File   ：pytorch_learn -> dataloader
@IDE    ：PyCharm
@Author ：Zhuge hammer
@Date   ：2024/1/23 9:33
@Desc   ：
*************************************************
"""
import  torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10('./dataset', train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

writer = SummaryWriter('./dataloader')
step = 0
for data in test_loader:
    imgs, targets = data

    writer.add_images('dataloader_drop', imgs, step)
    step += 1

writer.close()