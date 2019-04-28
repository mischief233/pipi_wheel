# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 20:20:02 2019

@author: wayne
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import datetime
import time
import warnings
warnings.filterwarnings('ignore')

grouded = pd.read_csv('66666.csv')

#grouded= grouded.drop(grouded.columns[[2,3,9,10]], axis=1)

#最大值减中位数
def max_median(arr):
    return arr.max()-arr.median()
b = grouded.groupby(['uid','worldid']).agg(max_median)

b.to_csv("777777.csv",sep=',')