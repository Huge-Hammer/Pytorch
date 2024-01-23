#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
*************************************************
@Project -> File   ：pytorch_learn -> transform_learn
@IDE    ：PyCharm
@Author ：Zhuge hammer
@Date   ：2024/1/22 17:10
@Desc   ：
*************************************************
"""
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

# 示例图片
img_path = 'dataset/train/ants/0013035.jpg'
img = Image.open(img_path)
writer = SummaryWriter('logs')

# transforms如何使用
# 常用的transforms

# 1. tensor数据类型
# 函数实例化
trans_totensor = transforms.ToTensor()
# 转换
tensor_img = trans_totensor(img)

writer.add_image('tensor_img', tensor_img)

# 2. Normalize
print(tensor_img[0][0][0])  # tensor_img的第一个通道的第一个像素点的值
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 三个通道的均值和方差
norm_img = trans_norm(tensor_img)
print(norm_img[0][0][0])    # 归一化后的第一个通道的第一个像素点的值

writer.add_image('norm_img', norm_img)

# 3. Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
resize_img = trans_resize(img)       # PIL to PIL
resize_img = trans_totensor(resize_img)     # PIL to tensor
writer.add_image('resize_img', resize_img,0)
print(resize_img.size)

# 4. Compose
trans_resize2 = transforms.Resize(1024)
trans_compose = transforms.Compose([trans_resize2, trans_totensor])
resize_compose = trans_compose(img)
writer.add_image('resize_img', resize_compose, 1)

# 5. RandomCrop
trans_random = transforms.RandomCrop(100)
trans_compose = transforms.Compose([trans_random, trans_totensor])
crop_compose = trans_compose(img)
writer.add_image('crop_img', crop_compose)

writer.close()


