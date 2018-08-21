#encoding:utf-8
import csv
from numpy import *
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

training_set_inputs = []
training_set_outputs = []

# 读取训练集
csv_file = csv.reader(open('train.csv', 'r'))
# 收集数据
for data in csv_file:
    training_set_inputs += [data]
    training_set_outputs.append(data[1])
# 清除首部标题字段
training_set_outputs.pop(0)
training_set_inputs.pop(0)
# 数据转换为矩阵
training_set_inputs = array(training_set_inputs)
training_set_outputs = array(training_set_outputs).T
# 清除不必要的特征
training_set_inputs = delete(training_set_inputs, [0, 1, 3, 8], axis=1)
# 设置初始权重m
random.seed(1)
synaptic_weights = 2 * random.random((9, 1)) - 1
# 训练模型

# print(training_set_inputs)




