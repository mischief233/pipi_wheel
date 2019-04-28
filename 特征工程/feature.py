# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 04:42:33 2019

@author: wayne
"""

#导入库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import datetime
import time
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set_style('white')

all = pd.read_csv(r'C:/Users/wayne/Desktop/contest/train_feature.csv')
#一轮选择
all1 = all.drop(all.columns[[]], axis=1)
all1 = all1[all1['ext_flag']!='0x400000']
all1 = all1[all1['iLeaver']!=1]
all1 = all1[all1['ext_flag']!='0x20']
all1 = all1[all1['uid']!=0]
del all1['iLeaver']

#数据类型
all1.loc[all1['Elo_change'] =='\\N'] = 0
all1['Elo_change'] = all1['Elo_change'].astype("int")
all1.info()

#分类型
fl = all1.iloc[:,[-2,-1,6,5]]
fl.ix[(fl['flag']!='0x1')&(fl['flag']!='0x2')&(fl['flag']!='0x4')&(fl['flag']!='0x8')&(fl['flag']!='0x1')&(fl['flag']!='0x10'),'flag']='else'
fl.ix[(fl['ext_flag']!='0x1')&(fl['ext_flag']!='0x2')&(fl['ext_flag']!='0x4')
&(fl['ext_flag']!='0x8')&(fl['ext_flag']!='0x10')&(fl['ext_flag']!='0x40')&(fl['ext_flag']!='0x80')
&(fl['ext_flag']!='0x100')&(fl['ext_flag']!='0x200')&(fl['ext_flag']!='0x400')&(fl['ext_flag']!='0x800')
&(fl['ext_flag']!='0x1000')&(fl['ext_flag']!='0x2000')&(fl['ext_flag']!='0x4000')&(fl['ext_flag']!='0x8000')
&(fl['ext_flag']!='0x10000')&(fl['ext_flag']!='0x20000')&(fl['ext_flag']!='0x40000')&(fl['ext_flag']!='0x80000')
&(fl['ext_flag']!='0x100000')&(fl['ext_flag']!='0x200000')&(fl['ext_flag']!='0x800000'),'ext_flag'] = 'else'

#独热编码
fl=pd.get_dummies(fl)

#计数
flags = fl.groupby(['uid','worldid']).sum()

#id
ids = all1.iloc[:,[-2,-1]]

#数值型
sz = all1.drop(all1.columns[[2,-4,6,5]], axis=1)
#统计特征
function = ['mean','var']
a = sz.groupby(['uid','worldid']).agg(function)
#最大值减中位数
def max_median(arr):
    return arr.max()-arr.median()
b = sz.groupby(['uid','worldid']).agg(max_median)

#上线次数
c = sz.groupby(['uid','worldid']).uid.count()
c = pd.DataFrame(c)
c.rename(columns={'uid':'count'}, inplace = True)
d = pd.merge(flags,c,left_index=True,right_index=True,how='inner')

#比率
name = [ x for x in fl.columns]
name1 = [x+'_rat' for x in name ]
for i in range(2,29):
    d[name1[i]] = d[name[i]]/d['count']
#差分特征

#H特征

#交叉特征
jc = all1.loc[:,['uid','worldid','game_score','champions_killed','iDuration','iMoneyChange','iRankInGame','iTurretsDestroyed','Elo_change']]

d.to_csv("C:/Users/wayne/Desktop/contest/new_flags.csv",sep=',')
