# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 12:51:36 2018

@author: jinqiu
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'C:/Users/jinqiu/Desktop/train.csv', encoding = "gbk")
#test = pd.read_csv(r'C:/Users/jinqiu/Desktop/test.csv', encoding = "gbk")

train = dataset.iloc[:,:-1]
label = dataset.iloc[:,-1].values

#数据标准化
scaler = MinMaxScaler()
train = scaler.fit(train).transform(train)
#test = scaler.fit(test).transform(test)

#训练分类器
clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=4, max_features='auto', max_leaf_nodes=None, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=9,
            min_weight_fraction_leaf=0.0, n_estimators=140, n_jobs=1,
            oob_score=False, random_state=14, verbose=0, warm_start=False)
clf = clf.fit(train, label)
#预测结果
#pre = clf.predict(test)

#重要性排序
importances = list(clf.feature_importances_)
names = list(data.columns)
names.remove('label')
def list_add(a,b):
    all = []
    for i in range(len(a)):
        all.append([a[i],b[i]])
    return all
all = list_add(names,importances)
all.sort(key=lambda x:x[1],reverse = True)
all
