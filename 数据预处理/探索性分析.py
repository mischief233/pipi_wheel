# -*- coding: utf-8 -*-
"""
Created on Fri May 18 19:20:56 2018

@author: jinqiu
"""


import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

creditdata = pd.read_excel('C:/Users/wayne/Desktop/creditdata.xlsx')
del creditdata['ID']

lable = creditdata.iloc[:,-1]
del creditdata['Class']
from sklearn import preprocessing
data1 = preprocessing.StandardScaler().fit(creditdata).transform(creditdata)
data2 = pd.DataFrame(data1,columns=creditdata.columns)
train = pd.concat([data2,lable],axis=1)

train = creditdata.iloc[0:284726,:]
test = creditdata.iloc[284727:284807,:]

plt.figure(figsize=(12,5))
plt.subplot(121)
creditdata['V1'].hist(bins=70)
plt.xlabel('V1')
plt.ylabel('Num')

plt.subplot(122)
train.boxplot(column='V1', showfliers=False)
plt.show()

facet = sns.FacetGrid(train, hue="Class",aspect=3)
facet.map(sns.kdeplot,'Amount',shade= True)
facet.set(xlim=(min(), train['Amount'].max()))
facet.add_legend()

creditdata['Fare_bin'] = pd.qcut(creditdata['Fare'], 5)
creditdata['Fare_bin'].head()

train.to_csv('C:/Users/wayne/Desktop/train.csv')
test.to_csv('C:/Users/wayne/Desktop/test.csv')