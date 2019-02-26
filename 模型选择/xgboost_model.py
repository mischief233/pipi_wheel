# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 16:16:45 2018

@author: jinqiu
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt

#数据导入、检查空缺值
data = pd.read_csv(r'C:/Users/jinqiu/Desktop/train.csv', encoding = "gbk")
train = data.iloc[:,:-1]
label = data.iloc[:,-1]

#数据标准化
scaler = MinMaxScaler()
train = scaler.fit(data).transform(data)

# 训练模型
clf = xgb.XGBClassifier(random_state=29,
                        n_jobs=-1,
                        max_depth=18,
                        n_estimator=55,
                        learning_rate=0.05,
                        subsample=0.7,
                        min_child_weight=1)
f1 = cross_val_score(clf, train, label, scoring='f1',cv=5)
print("f1:{0:.1f}%".format(np.mean(f1)*100))

# 显示重要特征
clf.fit(train, label)
fig, ax = plt.subplots(figsize=(5, 10))
plot_importance(clf, ax=ax)
plt.show()

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
