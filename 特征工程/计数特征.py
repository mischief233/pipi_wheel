# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 17:49:42 2018

@author: jinqiu
"""
import numpy as np
import pandas as pd

bleningdata= dataset
bleningdata['n1'] = (bleningdata==1).sum(axis=1)
bleningdata['n2'] = (bleningdata==2).sum(axis=1)
bleningdata['n3'] = (bleningdata==3).sum(axis=1)
bleningdata['n4'] = (bleningdata==4).sum(axis=1)
bleningdata['n5'] = (bleningdata==5).sum(axis=1)
bleningdata.drop(bleningdata.columns[0:2], axis=1,inplace=True)
del creditdata['Class']
data1 = []
from sklearn.preprocessing import PolynomialFeatures
del creditdata['V4']
data = creditdata.values
for i in range(27):
    for j in range(i+1,27,1):
        a = data[:,[i,j]]
        poly = PolynomialFeatures(degree=2, interaction_only=True)
        data1.append(poly.fit_transform(a))
data2 = np.array(data1)[0][:,1:4]
print(data1[0][:,1:4])
data3 = []
data3 = np.array(data3)
for t in range(350):
    data2 = np.column_stack((data2,data1[t+1][:,1:4]))
data2 = pd.DataFrame(data2)