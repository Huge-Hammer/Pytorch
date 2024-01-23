#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
*************************************************
@Project -> File   ：pytorch_learn -> tb_learn
@IDE    ：PyCharm
@Author ：Zhuge hammer
@Date   ：2024/1/22 16:14
@Desc   ：learn tensor board
*************************************************
"""
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter('logs')
img_path = "dataset/train/ants/0013035.jpg"
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)

writer.add_image("train", img_array, 1, dataformats='HWC')
writer.add_scalar('loss', 0.15, 1)
writer.close()