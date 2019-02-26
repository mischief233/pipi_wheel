# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 18:15:23 2018

@author: jinqiu
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

#数据导入、检查空缺值
data = pd.read_csv('C:/Users/jinqiu/Desktop/train.csv', encoding = "gbk")
train = data.iloc[:,:-1]
label = data.iloc[:,-1]

#数据标准化
scaler = MinMaxScaler()
train = scaler.fit(train).transform(train)

#train_data = lgb.Dataset(train, label=label)
#数据集拆分
train, val, train_label, val_label = train_test_split(train, label, test_size=0.2)
val_label = np.array(val_label)
#数据转换
lgb_train = lgb.Dataset(train, train_label, free_raw_data=False)
lgb_eval = lgb.Dataset(val, val_label, reference=lgb_train,free_raw_data=False)

### 开始训练
params = {
             'boosting_type': 'gbdt',
             'boosting': 'dart',
             'objective': 'binary',
             'metric': 'binary_logloss',
 
             'learning_rate': 0.01,
             'num_leaves':25,
             'max_depth':3,
 
             'max_bin':10,
             'min_data_in_leaf':8,
 
             'feature_fraction': 0.6,
             'bagging_fraction': 1,
             'bagging_freq':0,
 
             'lambda_l1': 0,
             'lambda_l2': 0,
             'min_split_gain': 0
}
#num_round=10
#sco = lgb.cv(params, train_data, num_round, nfold=5)
gbm = lgb.train(params,                     # 参数字典
                 lgb_train,                  # 训练集
                 num_boost_round=100,       # 迭代次数
                 valid_sets=lgb_eval,        # 验证集
                 early_stopping_rounds=10)   # 早停系数
val_pre = gbm.predict(val, num_iteration=gbm.best_iteration) 

f1 = f1_score(val_label, np.sign(val_pre))
print("f1:{0:.1f}%".format(f1*100))


