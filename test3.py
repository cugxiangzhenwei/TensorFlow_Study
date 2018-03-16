# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#使用NumPy生成假数据,总共100个点
x_data = np.float32(np.random.rand(2,100)) #随机输入100个点
y_data = np.dot([0.100,0.200],x_data) + 0.300

print x_data
print y_data
