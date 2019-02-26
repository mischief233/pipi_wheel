# -*- coding: utf-8 -*-
"""
Created on Fri May 18 12:32:31 2018

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

creditdata = pd.read_excel('C:/Users/wayne/Desktop/creditdata.xlsx')
del creditdata['ID']

creditdatacopy = creditdata.copy()


lable = creditdata.iloc[:,-1]
del creditdata['Class']
x=creditdata.values
x[:,[1,3]]

sns.set_style('whitegrid')
creditdata.head()
creditdata.info()
creditdata.tail()
creditdata.describe()
lable.value_counts()

x = creditdatacopy['V2']
binarizer = preprocessing.Binarizer(threshold=1.8)
creditdata['V2'] = binarizer.transform(x).T

lable = creditdata.iloc[:,-1]
del creditdata['Class']
del creditdata['V2new']

data1 = preprocessing.StandardScaler().fit(creditdata).transform(creditdata)
data2 = pd.DataFrame(data1,columns=creditdata.columns)
creditdata = pd.concat([data2,lable],axis=1)
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
facet.set(xlim=(-10, train['Amount'].max()))
facet.add_legend()


coumle =['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Time','Amount']
for i in coumle:
    group_name=[1,2,3,4,5]
    creditdata[i]= pd.qcut(creditdata[i],5,labels=group_name)
    
g = sns.pairplot(train[[u'V1', u'V2', u'V3', u'V4', u'V5']], palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )
g.set(xticklabels=[])


creditdata.to_csv('C:/Users/wayne/Desktop/blening.csv')
train.to_csv('C:/Users/wayne/Desktop/train.csv')
test.to_csv('C:/Users/wayne/Desktop/test.csv')


