# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:14:31 2019

@author: wayne
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import time
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set_style('white')


#导入
df = pd.read_csv(r'C:/Users/wayne/Desktop/contest/train1.txt')
all1 = pd.read_csv(r'C:/Users/wayne/Desktop/contest/all1.csv')

train = pd.merge(all1,ids,on=['uid','worldid'],how='inner')

ab = train[train['label']==1]
jc = ab.loc[:,['game_score','Elo_change']]
jc.reset_index(inplace = True)
jc.set_index(['uid','worldid'], inplace = True)
from sklearn.cluster import KMeans
estimator = KMeans(n_clusters=2).fit(jc)
label_pred = estimator.labels_ #获取聚类标签
centroids = estimator.cluster_centers_ #获取聚类中心
inertia = estimator.inertia_ # 获取聚类准则的总和

#del jc['yp']
label_pred = pd.DataFrame(label_pred,names = ['fs'])
res = jc.append()
jc['yp'] = label_pred.values

ab['yp'] = label_pred.values

ab1 = ab[ab['yp']==1]
ab2 = jc[jc['yp']==1]

ab1.to_csv("C:/Users/wayne/Desktop/contest/new_ab1.csv",index=False,sep=',')

nor = ids[ids['label']!=1]
train = train[train['label']==0]
nor1 = train[train['iDamageTaken']<30000]
nor1 = train[train['iGoldEarned']<20000]

nor1.to_csv("C:/Users/wayne/Desktop/contest/new_nor1.csv",index=False,sep=',')
ab1 = pd.read_csv(r'C:/Users/wayne/Desktop/contest/new_ab1.csv')
alldata = pd.concat([nor1,ab1],axis = 0)
del alldata['yp']
alldata.info()

label1 = all.loc[:,'uid','worldid','label']
train_feature.to_csv("C:/Users/wayne/Desktop/contest/train_feature.csv",index=False,sep=',')
target.to_csv("C:/Users/wayne/Desktop/contest/target999.csv",index=False,sep=',')
