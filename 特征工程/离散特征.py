# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 16:11:59 2018

@author: jinqiu
"""
import pandas as pd
import numpy as np

#数据集名称dataset
scatter = dataset.ix[:,['无功功率控制状态','风机当前状态值','轮毂当前状态值','偏航状态值','偏航要求值']]
#独热编码
new_scatter = pd.get_dummies(scatter,columns=['无功功率控制状态','风机当前状态值','轮毂当前状态值','偏航状态值','偏航要求值'])
#计数
scatter_feature = new_scatter.sum(axis=0)