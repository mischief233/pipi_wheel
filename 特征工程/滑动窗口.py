# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 09:40:27 2019

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

#导入
all = pd.read_csv(r'C:/Users/wayne/Desktop/contest/all2.csv')
all.info()

all1 = all.drop(all.columns[[2,9,10]], axis=1)
all1.info()

all1[ 'battle_time'] = pd.to_datetime(all1['battle_time'])

all1.set_index('battle_time', inplace = True)
all1.index=all1.index.droplevel(level=2)
all1.head()

a = df['2019-03-07']
grouded = a.groupby(['uid', 'worldid']).rolling(window = 3,axis =1).mean()
del grouded['uid']
del grouded['worldid']

grouded.index=grouded.index.droplevel(level=2)

grouded = pd.read_csv(r'C:/Users/wayne/Desktop/contest/grouded.csv')
b = grouded.groupby(['uid', 'worldid']).var()

c = pd.read_csv(r'C:/Users/wayne/Desktop/contest/hd_result.csv')
grouded.to_csv("C:/Users/wayne/Desktop/contest/grouded.csv",sep=',')
