# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 13:13:55 2018

@author: jinqiu
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from time import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost.sklearn import XGBClassifier
from matplotlib import pyplot as plt

#数据导入、检查空缺值
data = pd.read_csv(r'C:/Users/jinqiu/Desktop/data.csv',encoding = "gbk")
data.info()
data.notnull().sum(axis=0)/data.shape[0]
train = data.iloc[:,:-1]
label = data.iloc[:,-1]

#数据标准化
scaler = MinMaxScaler()
train = scaler.fit(train).transform(train)

#定义分类器
alg1=DecisionTreeClassifier(random_state=29)
alg2=SVC(probability=True,random_state=29)  #由于使用roc_auc_score作为评分标准，需将SVC中的probability参数设置为True
alg3=RandomForestClassifier(random_state=29)
alg4=AdaBoostClassifier(random_state=29)
alg5=KNeighborsClassifier(n_jobs=-1)
alg6=XGBClassifier(random_state=29,n_jobs=-1)

#定义调参函数
def fit_model(alg,parameters):
    grid = GridSearchCV(alg,parameters,scoring='f1',cv=5)  #使用网格搜索，出入参数
    start=time()  #计时
    grid=grid.fit(train,label)  #模型训练
    end=time()
    t=round(end-start,3)
    print (grid.best_params_ ) #输出最佳参数
    print ('searching time for {} is {} s'.format(alg.__class__.__name__,t)) #输出搜索时间
    return grid #返回训练好的模型

#列出需要调整的参数范围
parameters1={'max_depth':range(1,10),'min_samples_split':range(2,10)}
parameters2 = {"C":range(1,20), "gamma": [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]}
parameters3_1 = {'n_estimators':range(10,200,10)}
parameters3_2 = {'max_depth':range(1,10),'min_samples_split':range(2,10)}  
parameters4 = {'n_estimators':range(10,200,10),'learning_rate':[i/10.0 for i in range(5,15)]}
parameters5 = {'n_neighbors':range(2,10),'leaf_size':range(10,80,20)  }
parameters6_1 = {'n_estimators':range(10,200,10)}
parameters6_2 = {'max_depth':range(1,10),'min_child_weight':range(1,10)}
parameters6_3 = {'subsample':[i/10.0 for i in range(1,10)], 'colsample_bytree':[i/10.0 for i in range(1,10)]}

#开始调参
clf1=fit_model(alg1,parameters1)

clf2=fit_model(alg2,parameters2)

clf3_m1=fit_model(alg3,parameters3_1)

alg3=RandomForestClassifier(random_state=29,n_estimators=180)
clf3=fit_model(alg3,parameters3_2)

clf4=fit_model(alg4,parameters4)

clf5=fit_model(alg5,parameters5)

clf6_m1=fit_model(alg6,parameters6_1)

alg6=XGBClassifier(n_estimators=140,random_state=29,n_jobs=-1)
clf6_m2=fit_model(alg6,parameters6_2)

alg6=XGBClassifier(n_estimators=140,max_depth=4,min_child_weight=5,random_state=29,n_jobs=-1)
clf6=fit_model(alg6,parameters6_3)

#定义交叉验证
def cross_val(clf,train,label):
    f1 = cross_val_score(clf, train, label, scoring='f1')
    return np.mean(f1)*100

#初始化结果列表
results = []

#循环所有模型
models = [clf1,clf2,clf3,clf4,clf5,clf6]
for i in models:
    f1 = cross_val(i,train,label)
    results.append(f1)
    
name_list = ['DecisionTree','SVM','RandomForest','Adaboost','KNN','XGBOOST']
plt.barh(range(len(results)), results, color='rgb',tick_label=name_list)
plt.show()
    









