# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 15:32:46 2018

@author: jinqiu
"""

import pandas as pd
import numpy as np

data = pd.read_csv(r'C:/Users/jinqiu/Desktop/train-csvdata/testdata/discrete.csv', encoding = "gbk")
data_name = pd.read_csv('C:/Users/jinqiu/Desktop/train.csv', encoding = "gbk")

data_name = data_name.drop(['大气压力','label'],axis=1)

train_mean.columns = data_name.columns
train_mean = train_mean.drop(['无功功率控制状态','风机当前状态值','轮毂当前状态值','偏航状态值','偏航要求值'], axis=1)
train_mean.to_csv(r'C:/Users/jinqiu/Desktop/new-train/count_0.csv',index= False)