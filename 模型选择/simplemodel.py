# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 13:26:14 2018

@author: jinqiu
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

#数据导入、检查空缺值
data = pd.read_csv('C:/Users/jinqiu/Desktop/train.csv', encoding = "gbk")
data.info()
data.notnull().sum(axis=0)/data.shape[0]
train = data.iloc[:,:-1]
label = data.iloc[:,-1]

#数据标准化
scaler = MinMaxScaler()
train = scaler.fit(train).transform(train)

#单个分类器
clf = RandomForestClassifier(random_state=14)
f1 = cross_val_score(clf, train, label, scoring='f1')
print("f1:{0:.1f}%".format(np.mean(f1)*100))

#调参
parameter_space = {
        'n_estimators':range(10,200,10),
        'max_depth':range(1,10),
        'min_samples_split':range(2,10),
        }
clf = RandomForestClassifier(random_state=14)
grid = GridSearchCV(clf,parameter_space,scoring='f1')
grid.fit(train,label)
print("f1:(0:.1f)%".format(grid.best_score_*100))
print(grid.best_estimator_)

#调参后的分类器
new_clf = RandomForestClassifier(random_state=20, n_estimators=10, max_depth=None, min_samples_split=2)
f1 = cross_val_score(new_clf, train, label, scoring='f1')
print("f1:{0:.1f}%".format(np.mean(f1)*100))