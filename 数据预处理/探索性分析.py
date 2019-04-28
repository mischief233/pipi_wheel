# -*- coding: utf-8 -*-
"""
Created on Fri May 18 19:20:56 2018

@author: jinqiu
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
import csv

#导入
df = pd.read_csv(r'C:/Users/wayne/Desktop/contest/all2.csv' )
train_label = pd.read_csv(r'C:/Users/wayne/Desktop/contest/train_label.csv')
all = pd.read_csv(r'C:/Users/wayne/Desktop/contest/alldata1.txt')
test = pd.read_table(r'C:/Users/wayne/Desktop/contest/test.txt',sep='|',names=['uid','worldid'], header=None)

#数据类型
df.loc[df['Elo_change'] =='\\N'] = 0
df['Elo_change'] = df['Elo_change'].astype("int")

ab = df[df['label']==1]
#比例
a = c['label'].value_counts()
bili = a[1]/a[0]

df.head()
del df['game_mode']

#绘图
sns.countplot(x='Premade_size',data = df)
plt.savefig('ex_flag.png')

sns.distplot(ab['Elo_change'])

sns.boxplot(ab['Elo_change'])

facet = sns.FacetGrid(df, hue="label",aspect=4)
facet.map(sns.kdeplot,'Elo_change',shade= True)
facet.set(xlim=(0, df['Elo_change'].max()))
facet.add_legend()

b = df[df['Premade']!=false]
c= train[train['Elo_change']>30]
d= ab[ab['Elo_change']>30]
e = ab['ext_flag'].value_counts().rank()

#格式转换
df['flag'] = pd.DataFrame(list(map(lambda x: hex(x),df['flag'])))
all['flag'] = pd.DataFrame(list(map(lambda x: hex(x),all['flag'])))
df['ext_flag'] = pd.DataFrame(list(map(lambda x: hex(x),df['ext_flag'])))
all['ext_flag'] = pd.DataFrame(list(map(lambda x: hex(x),all['ext_flag'])))


ab.to_csv("C:/Users/wayne/Desktop/contest/ab.txt",index=False,sep=',')
df.to_csv("C:/Users/wayne/Desktop/contest/train1.txt",index=False,sep=',')
all.to_csv("C:/Users/wayne/Desktop/contest/alldata1.txt",index=False,sep=',')