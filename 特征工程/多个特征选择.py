# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 16:04:54 2018

@author: jinqiu
"""

import pandas as pd
import numpy as np
from feature_selector import FeatureSelector
import glob

path = glob.glob(r'C:/Users/jinqiu/Desktop/new-train/*.csv')
'''d=[]
for item in path:
    a=item.split('\\')
    hh=a[-1].split('.')
    d.append(hh[0])
print(d)'''
label = pd.read_csv(r'C:/Users/jinqiu/Desktop/select_data/new_label.csv', encoding = "gbk")
for i in path:
    a=i.split('\\')
    hh=a[-1].split('.')
    train = pd.read_csv(i, encoding = "gbk")
    fs = FeatureSelector(data = train, labels = label)
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
    train_removed = train.drop(list1, axis=1)
    file_handle =open('C:/Users/jinqiu/Desktop/remove/'+hh[0]+'.txt',mode='w')
    for name in list1:
        file_handle.write(name)
        file_handle.write('\n')
    file_handle.close()
    train_removed.to_csv(r'C:/Users/jinqiu/Desktop/dataset/'+hh[0]+'.csv',index=False)