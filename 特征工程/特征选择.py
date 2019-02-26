# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 14:35:36 2018

@author: jinqiu
"""
import pandas as pd
import numpy as np
from feature_selector import FeatureSelector

data = pd.read_csv(r'C:/Users/jinqiu/Desktop/new-train/train_mean.csv', encoding = "gbk")
train = all
label = pd.read_csv(r'C:/Users/jinqiu/Desktop/select_data/new_label.csv', encoding = "gbk")
#train = data.iloc[:,:-1]
#label = data.iloc[:,-1]
#train.columns = ['F'+str(i+1) for i in range(125)]

fs = FeatureSelector(data = train, labels = label)

#共线性特征（线性线性相关）
fs.identify_collinear(correlation_threshold = 0.98)
collinear_features = fs.ops['collinear']
fs.record_collinear

#零重要度、重要性排序
fs.identify_zero_importance(task = 'classification', 
 eval_metric = 'auc', 
 n_iterations = 10, 
 early_stopping = True)
zero_importance_features = fs.ops['zero_importance']
fs.plot_feature_importances(threshold = 0.99, plot_n = 30)

#低重要度特征
fs.identify_low_importance(cumulative_importance = 0.99)
fs.feature_importances

#单个唯一值特征
fs.identify_single_unique()
fs.plot_unique()

remove = []
remove.extend(fs.ops['collinear'])
remove.extend(fs.ops['low_importance'])
remove.extend(fs.ops['zero_importance'])
remove.extend(fs.ops['single_unique'])
list1 = list(set(remove))
#移除特征
train_removed = train.drop(list1, axis=1)
#['missing', 'single_unique', 'collinear', 'zero_importance', 'low_importance'] methods have been run

fs.identify_all(selection_params = {'missing_threshold': 0.6, 
 'correlation_threshold': 0.8, 
 'task': 'classification', 
 'eval_metric': 'auc', 
 'cumulative_importance': 0.99})
file_handle =open('C:/Users/jinqiu/Desktop/remove/removed.txt',mode='w')
for name in list1:
    file_handle.write(name)
    file_handle.write('\n')
file_handle.close()
train_removed.to_csv(r'C:/Users/jinqiu/Desktop/select_data/train_count_0.csv',index=False)
