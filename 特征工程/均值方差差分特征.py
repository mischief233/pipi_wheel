# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 14:57:37 2018

@author: jinqiu
"""

import numpy as np
import pandas as pd

#计算差分的绝对值
new_data = abs(data.diff())
#差分累加值
difference = np.array(new_data.sum(axis=0))
#差分均值
difference_mean = np.array(np.mean(new_data))
#差分方差
difference_var = np.array(np.var(new_data))