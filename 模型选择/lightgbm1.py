# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 09:44:41 2019

@author: wayne
"""

# coding: utf-8
# pylint: disable = invalid-name, C0111
 
# 函数的更多使用方法参见LightGBM官方文档：http://lightgbm.readthedocs.io/en/latest/Python-Intro.html
 
import json
import numpy as np
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import  make_classification
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

data1 = pd.read_table(r'C:/Users/wayne/Desktop/contest/train_feature.csv',sep = ',')
data1 = data1.fillna(0)

data=data1.drop(data1.columns[[0,1,-1]], axis=1)
target = data1['label']
X_train,X_test,y_train,y_test =train_test_split(data,target,test_size=0.2)
 
# 加载你的数据
# print('Load data...')
# df_train = pd.read_csv('../regression/regression.train', header=None, sep='\t')
# df_test = pd.read_csv('../regression/regression.test', header=None, sep='\t')
#
# y_train = df_train[0].values
# y_test = df_test[0].values
# X_train = df_train.drop(0, axis=1).values
# X_test = df_test.drop(0, axis=1).values
 
# 创建成lgb特征的数据集格式
lgb_train = lgb.Dataset(X_train, y_train) # 将数据保存到LightGBM二进制文件将使加载更快
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)  # 创建验证数据
 
# 将参数写成字典下形式
params = {
    'task': 'train',
    'metric': {'l2', 'auc'},  # 评估函数
    'boosting_type': 'gbdt', 
    'objective': 'regression', 
    'n_estimators': 120,
    'learning_rate': 0.1, 
    'num_leaves': 30, 
    'max_depth': 6,   
    'subsample': 0.8, 
    'colsample_bytree': 0.8,
     'is_unbalance':True,
}
 
print('Start training...')
# 训练 cv and train
num_round = 10
lgb.cv(params, lgb_train, num_round, nfold=5)
gbm = lgb.train(params,lgb_train,num_boost_round=1000,valid_sets=lgb_eval,early_stopping_rounds=5) # 训练数据需要参数列表和数据集
 
print('Save model...') 
 
gbm.save_model('model.txt')   # 训练后保存模型到文件

print('Start predicting...')

'''
data1 = pd.read_table(r'C:/Users/wayne/Desktop/contest/train_feature.csv',sep = ',')
X_test = data1.drop(data1.columns[[0,1,-1]], axis=1)
y_test = data1['label']
'''

# 预测数据集
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration) #如果在训练期间启用了早期停止，可以通过best_iteration方式从最佳迭代中获得预测
# 评估模型
def arr(x):
    if x>0.5:
        return 1
    else:
        return 0
pre = [arr(x) for x in y_pred]
p = precision_score(y_test, pre)
r = recall_score(y_test, pre)
a = 4*p*r/(p+3*r)
print(a)
#print(a) # 计算真实值和预测值之间的均方根误差