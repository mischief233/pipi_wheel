# -*- coding: utf-8 -*-
"""
Created on Fri May 18 19:29:04 2018

@author: wayne
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

import warnings
warnings.filterwarnings('ignore')
creditdata = pd.read_csv('C:/Users/wayne/Desktop/creditdata.csv')
bleningdata = pd.DataFrame(bleningdata, dtype='float')
lable =bleningdata.iloc[:,-1]
del bleningdata['Time']
del bleningdata['Class']
creditdata = pd.DataFrame(creditdata, dtype='float')
#del bleningdata['Unnamed']
train = bleningdata.iloc[0:284726,:]
test = bleningdata.iloc[284727:284807,:]
bleningdata= creditdata

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

bleningdata.to_csv('C:/Users/wayne/Desktop/bleningdata3.csv')
data2.to_csv('C:/Users/wayne/Desktop/crossdata.csv')