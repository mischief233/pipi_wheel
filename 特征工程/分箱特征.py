# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 17:50:45 2018

@author: jinqiu
"""
import numpy as np
import pandas as pd

dataset.drop(dataset.columns[['无功功率控制状态','风机当前状态值','轮毂当前状态值','偏航状态值','偏航要求值']], axis=1,inplace=True)
