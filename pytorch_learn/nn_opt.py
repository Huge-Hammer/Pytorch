#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
*************************************************
@Project -> File   ：pytorch_learn -> nn_opt
@IDE    ：PyCharm
@Author ：Zhuge hammer
@Date   ：2024/1/23 16:42
@Desc   ：
*************************************************
"""
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class hammer(nn.Module):
    def __init__(self):
        super(hammer, self).__init__()
        self.conv1 = Conv2d(3, 32, 5, 1, 2)
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(32, 32, 5, 1, 2)
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = Conv2d(32, 64, 5, 1, 2)
        self.maxpool3 = MaxPool2d(2)
        self.flatten = Flatten()
        self.linear1 = Linear(1024, 64)
        self.linear2 = Linear(64, 10)

        self.model1 = Sequential(
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
        x = self.model1(x)
        return x

dataset = torchvision.datasets.CIFAR10(root='dataset', train=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

HA = hammer()
loss = nn.CrossEntropyLoss()
opt = torch.optim.SGD(HA.parameters(), lr=1e-3)

for epoch in range(20):          # 训练所有数据集20次
    run_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = HA(imgs)
        l = loss(outputs, targets)
        opt.zero_grad()
        l.backward()
        opt.step()
        run_loss += l.item()
    print("epoch:{}, loss:{}".format(epoch, run_loss))