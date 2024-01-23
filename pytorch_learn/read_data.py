#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
*************************************************
@Project -> File   ：pytorch_learn -> read_data
@IDE    ：PyCharm
@Author ：Zhuge hammer
@Date   ：2024/1/22 15:14
@Desc   ：learn load dataset
*************************************************
"""
from torch.utils.data import Dataset
from PIL import Image
import os


class MyDataset(Dataset):
    def __init__(self, rootdir, labeldir):
        self.rootdir = rootdir
        self.labeldir = labeldir
        self.path = os.path.join(self.rootdir, self.labeldir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, index):
        img_name = self.img_path[index]
        img_item_path = os.path.join(self.path, img_name)
        img = Image.open(img_item_path)
        label = self.labeldir
        return img, label

    def __len__(self):
        return len(self.img_path)

if __name__ == '__main__':
    rootdir = 'dataset/train'
    ant_dir = 'ants'
    bee_dir = 'bees'
    ant_dataset = MyDataset(rootdir, ant_dir)
    bee_dataset = MyDataset(rootdir, bee_dir)
    train_dataset = ant_dataset + bee_dataset
    print(len(train_dataset))



