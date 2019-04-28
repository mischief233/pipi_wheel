# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 09:02:08 2019

@author: wayne
"""

#导入库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import datetime
import time
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set_style('white')

#导入
a = pd.read_csv(r'C:/Users/wayne/Desktop/contest/day_mean_var.csv')
b = pd.read_csv(r'C:/Users/wayne/Desktop/contest/train_feature.csv')
c = pd.read_csv(r'C:/Users/wayne/Desktop/contest/max_median.csv')

function = ['_mean','_var']
name = [ x for x in sz.columns]
names = []
for i in name:
    for j in function:
        names.append(i+j)
names = names[:-4]
e = ['uid','worldid']
e.extend(names)
a.columns = e

c_name = [ x for x in c.columns]

d = pd.merge(a,b,on=['uid','worldid'],how='inner')
data1 = pd.merge(data1,ids,on=['uid','worldid'],how='right')
data1 = data1.fillna(0)

train_label = train_feature['label']
train_label.to_csv("C:/Users/wayne/Desktop/contest/train_label.csv",index=False,sep=',')


from feature_selector import FeatureSelector
data = data1.drop(data1.columns[[0,1,-1]], axis=1)
fs = FeatureSelector(data = data, labels = data1['label'])
fs.identify_all(selection_params = {'missing_threshold': 0.6, 
 'correlation_threshold': 0.8, 
 'task': 'classification', 
 'eval_metric': 'auc', 
 'cumulative_importance': 0.99})
remove = []
remove.extend(fs.ops['collinear'])
remove.extend(fs.ops['low_importance'])
remove.extend(fs.ops['zero_importance'])
remove.extend(fs.ops['single_unique'])
list1 = list(set(remove))
#移除特征
sm = data.drop(list1, axis=1)
    



