#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
*************************************************
@Project -> File   ：pytorch_learn -> model_action
@IDE    ：PyCharm
@Author ：Zhuge hammer
@Date   ：2024/1/23 19:33
@Desc   ：
*************************************************
"""
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from model import *
from torch.utils.data import DataLoader

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root='dataset', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root='dataset', train=False, transform=torchvision.transforms.ToTensor(), download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练集数据大小：", train_data_size)
print("测试集数据大小：", test_data_size)

# 准备数据集加载器
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True)

# 构建模型
HA = hammer()

# 构建损失函数
loss_fn = nn.CrossEntropyLoss()

# 构建优化器
learning_rate = 1e-3
optimizer = torch.optim.Adam(HA.parameters(), lr=learning_rate)

# 训练模型
# 训练计数器
total_train_step = 0
# 测试计数器
total_test_step = 0
# 训练轮数
epochs = 10

# tensorboard
writer = SummaryWriter('./logs_train')

for i in range(epochs):
    print(f"第{i + 1}轮训练开始".center(60, "-"))

    # 训练
    for data in train_loader:
        imgs, targets = data
        outputs = HA(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f"第{total_train_step}步的loss：{loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试
    total_test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            outputs = HA(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
    print("测试损失：", total_test_loss)
    total_test_step += 1
    writer.add_scalar("test_loss", total_test_loss, total_test_step)

    # 保存模型
    torch.save(HA, "hammer{}.pth".format(i + 1))
    print("模型已保存")

writer.close()
