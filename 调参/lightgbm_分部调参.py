# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 22:44:06 2019

@author: wayne
"""

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV  # Perforing grid search
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('train_new.csv')   # 读取数据
y = pd.read_csv('label_new.csv')   # 用pop方式将训练数据中的标签值y取出来，作为训练目标，这里的‘30’是标签的列名
col = train_data.columns   
x = train_data[col].values  # 剩下的列作为训练数据
train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.333, random_state=0)   # 分训练集和验证集
train = lgb.Dataset(train_x, train_y)
valid = lgb.Dataset(valid_x, valid_y, reference=train)


parameters = {
              'max_depth': [15, 20, 25, 30, 35],
              'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
              'feature_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
              'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
              'bagging_freq': [2, 4, 5, 6, 8],
              'lambda_l1': [0, 0.1, 0.4, 0.5, 0.6],
              'lambda_l2': [0, 10, 15, 35, 40],
              'cat_smooth': [1, 10, 15, 20, 35]
}
gbm = lgb.LGBMClassifier(boosting_type='gbdt',
                         objective = 'binary',
                         metric = 'auc',
                         verbose = 0,
                         learning_rate = 0.01,
                         num_leaves = 35,
                         feature_fraction=0.8,
                         bagging_fraction= 0.9,
                         bagging_freq= 8,
                         lambda_l1= 0.6,
                         lambda_l2= 0)
# 有了gridsearch我们便不需要fit函数
gsearch = GridSearchCV(gbm, param_grid=parameters, scoring='accuracy', cv=3)
gsearch.fit(train_x, train_y)

print("Best score: %0.3f" % gsearch.best_score_)
print("Best parameters set:")
best_parameters = gsearch.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
best_parameters.to_csv("pa.txt")