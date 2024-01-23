#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
*************************************************
@Project -> File   ：pytorch_learn -> dataset_trans
@IDE    ：PyCharm
@Author ：Zhuge hammer
@Date   ：2024/1/22 20:32
@Desc   ：学习dataset的使用，结合transforms
*************************************************
"""
import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_trans = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize(256)])


train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=dataset_trans, download=True)
test_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=dataset_trans, download=True)

writer = SummaryWriter('dataset_logs')
for i in range(10):
    img, label = train_set[i]
    writer.add_image('train_set', img, i)

writer.close()