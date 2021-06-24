#! /user/bin/env python
# -*- coding: utf-8 -*-
"""
@author: gingkg
@contact: sby2015666@163.com
@software: PyCharm
@project: AlphaZero_Gomoku_PaddlePaddle
@file: test.py
@date: 2021/3/4
@desc: 测试用文件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


conv1 = nn.Conv2d(4,16,3,1,1)
fc1 = nn.Linear(16*10*10, 2)

a = torch.rand(2,4,10,10)
print(a)

x = conv1(a)
print(x)
print(x.shape)
x = x.view(x.shape[0], -1)
y = fc1(x)
print(y)

print(F.softmax(y, dim=0))
print(torch.exp(F.log_softmax(y, dim=0)))

