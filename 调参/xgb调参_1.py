# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 17:39:43 2018

@author: jinqiu
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from time import time
from xgboost.sklearn import XGBClassifier

#数据导入、检查空缺值
data = pd.read_csv('C:/Users/jinqiu/Desktop/train_difference_mean.csv', encoding = "gbk")
train = data.iloc[:,:-1]
label = data.iloc[:,-1]

#数据标准化
scaler = MinMaxScaler()
train = scaler.fit(train).transform(train)
#调参
def fit_model(alg,parameters):
    grid = GridSearchCV(alg,parameters,scoring='f1',cv=5)  #使用网格搜索，出入参数
    start=time()  #计时
    grid=grid.fit(train,label)  #模型训练
    end=time()
    t=round(end-start,3)
    print (grid.best_params_ ) #输出最佳参数
    print ('searching time for {} is {} s'.format(alg.__class__.__name__,t)) #输出搜索时间
    return grid #返回训练好的模型

parameters6 = {'n_estimators':range(10,200,10),
                 'max_depth':range(1,10),
                 'min_child_weight':range(1,10)},
                 'subsample':[i/10.0 for i in range(1,10)],
                 'colsample_bytree':[i/10.0 for i in range(1,10)]}

alg6 = XGBClassifier(random_state=29,n_jobs=6)
clf6 = fit_model(alg6,parameters6)

#调参后的分类器
f1 = cross_val_score(clf6, train, label, scoring='f1')
print("f1:{0:.1f}%".format(np.mean(f1)*100))