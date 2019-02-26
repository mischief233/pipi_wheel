# -*- coding: utf-8 -*-
"""
Created on Fri May 18 16:51:36 2018

@author: wayne
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 18 16:35:37 2018

@author: wayne
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 18 14:09:06 2018

@author: jinqiu
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from random import shuffle
from xgboost.sklearn import XGBClassifier
from sklearn.cross_validation import cross_val_score
import pickle
import time
from hyperopt import fmin, tpe, hp,space_eval,rand,Trials,partial,STATUS_OK
import random

data = pd.read_csv(r'C:/Users/jinqiu/Desktop/select_data/train_mean.csv', encoding = "gbk").values
label = pd.read_csv(r'C:/Users/jinqiu/Desktop/new-train/trainY.csv',header = None, encoding = "gbk").values
labels = label.reshape((1,-1))
label = labels.tolist()[0]

minmaxscaler = MinMaxScaler()
attrs = minmaxscaler.fit_transform(data)

index = range(0,len(label))
random.shuffle(label)
trainIndex = index[:int(len(label)*0.7)]
print (len(trainIndex))
testIndex = index[int(len(label)*0.7):]
print (len(testIndex))
attr_train = attrs[trainIndex,:]
print (attr_train.shape)
attr_test = attrs[testIndex,:]
print (attr_test.shape)
label_train = labels[:,trainIndex].tolist()[0]
print (len(label_train))
label_test = labels[:,testIndex].tolist()[0]
print (len(label_test))
print (np.mat(label_train).reshape((-1,1)).shape)


def GBM(argsDict):
    max_depth = argsDict["max_depth"] + 5
    n_estimators = argsDict['n_estimators'] * 5 + 50
    learning_rate = argsDict["learning_rate"] * 0.02 + 0.05
    subsample = argsDict["subsample"] * 0.1 + 0.7
    min_child_weight = argsDict["min_child_weight"]+1
    print ("max_depth:" + str(max_depth))
    print ("n_estimator:" + str(n_estimators))
    print ("learning_rate:" + str(learning_rate))
    print ("subsample:" + str(subsample))
    print ("min_child_weight:" + str(min_child_weight))
    global attr_train,label_train

    gbm = xgb.XGBClassifier(nthread=4,    #进程数
                            max_depth=max_depth,  #最大深度
                            n_estimators=n_estimators,   #树的数量
                            learning_rate=learning_rate, #学习率
                            subsample=subsample,      #采样数
                            min_child_weight=min_child_weight,   #孩子数
                            max_delta_step = 10,  #10步不降则停止
                            objective="binary:logistic")

    metric = cross_val_score(gbm,attr_train,label_train,cv=5).mean()
    print (metric)
    return -metric

space = {"max_depth":hp.randint("max_depth",16),
         #"n_estimators":hp.quniform("n_estimators",100,1000,1),  #[0,1,2,3,4,5] -> [50,]
        #"learning_rate":hp.quniform("learning_rate",0.01,0.2,0.01),  #[0,1,2,3,4,5] -> 0.05,0.06
         #"subsample":hp.quniform("subsample",0.5,1,0.1),#[0,1,2,3] -> [0.7,0.8,0.9,1.0]
         #"min_child_weight":hp.quniform("min_child_weight",1,6,1), #
         
         #"max_depth":hp.randint("max_depth",15),
         "n_estimators":hp.randint("n_estimators",10),  #[0,1,2,3,4,5] -> [50,]
         "learning_rate":hp.randint("learning_rate",6),  #[0,1,2,3,4,5] -> 0.05,0.06
         "subsample":hp.randint("subsample",3),#[0,1,2,3] -> [0.7,0.8,0.9,1.0]
         "min_child_weight":hp.randint("min_child_weight",5), #
        }
algo = partial(tpe.suggest,n_startup_jobs=1)
best = fmin(GBM,space,algo=algo,max_evals=4)

print (best)
print (GBM(best))