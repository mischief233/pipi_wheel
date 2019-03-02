#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 19:10:28 2019

@author: wayne
"""

#导入库
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import random
import time
import warnings
warnings.filterwarnings('ignore')
sns.set_style('white')

#数据导入
data = pd.read_csv('./filepath.csv',sep='', header=0(None))
#names列名list, index_col行索引label, dtype每列数据的数据类型 {‘a’: np.float64, ‘b’: np.int32}
#skiprows跳过的行号列表（从0开始), skipfooter从文件尾部开始忽略, nrows读取的行int, na_values替换NA/NaN的值
#encoding指定字符集类型'utf-8'gbk’
#data.to_csv("test.csv",index=False,sep=',')

'''csv.reader读写csv文件'''
import csv
with open("test.csv","r") as csvfile:
    reader = csv.reader(csvfile)
    #这里不需要readlines
    for line in reader:
        print (line)

#预览数据
data.head()
data.info()
data.tail()
data.describe()
data['label'].value_counts()
#直方图
data.hist(bins=50, figsize=(20,15)) 
plt.show
#and
sns.distplot(df['a']) 
plt.show()

#空缺值填补
#空缺值比例(axis=0行，1列
data.isnull().sum()
data.notnull().sun(axis=0)/data.shape[0]
#删除
del data['a']
data.drop(['a','b',axis=1], inplace = True)
#常规填补
data['a'].fillna('0', inplace = True)
data['a'].fillna(data['a'].mean(), inplace = True)
#中位数median(),众数model()[0]，插值interpolate()
#高级补法
data['a'].groupby(data['b']).median()
for i in range(len(data['a'])):
    if np.isnan(data.iloc[i,4]):
        if train.iloc[i,1]==1:
            train.iloc[i,4]=37
        elif train.iloc[i,1]==1:
            train.iloc[i,4]=29
        else:
            train.iloc[i,4]=24
#使用索引
b2 = data.groupy('a').b.median()
data.set_index('a', inplace = True)
data.b.fillna(b2, inplace = True)
data.reset_index(inplace = True) 

 b2 = data.groupy({'a','c'}).b.median()
data.set_index({'a','c'}, inplace = True)
data.b.fillna(b2, inplace = True)
data.reset_index(inplace = True)
          
#异常值
f,ax=plt.subplots(figsize=(10,8))
sns.boxplot(y='length',data=data,ax=ax)
plt.show()
#描述性统计
neg_list = ['销售数量', '应收金额', '实收金额']
for item in neg_list:
     neg_item = data[item]<0
     print(item + '小于0的有' + str(neg_item.sum()) + '个')
for item in neg_list:
     for i in range(0, len(data)):
         if data[item][i]<0:
             data = data.drop(i)
     neg_item = data[item]<0
     print(item + '小于0的有' + str(neg_item.sum()) + '个')
#三西格玛
for item in neg_list:
     data[item + '_zscore'] = (data[item] - data[item].mean())/data[item].std()
     z_abnormal = abs(data[item + '_zscore'])>3
     print(item + '中有' + str(z_abnormal.sum()) + '个异常值')
#箱型图
for item in neg_list:
     iqr = data[item].quantile(0.75) - data[item].quantile(0.25)
     q_abnormal_L = data[item] < data[item].quantile(0.25) - 1.5 * iqr
     q_abnormal_U = data[item] > data[item].quantile(0.75) + 1.5 * iqr
     print(item + '中有' + str(q_abnormal_L.sum() + q_abnormal_U.sum()) + '个异常值')
     
#独热编码
data=pd.get_dummies(data)
#相关特征转换成类别标签
from sklearn.preprocessing import LabelEncoder
dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])

#标准化或平均去除和方差缩放
from sklearn.preprocessing import MinMaxScaler
data = newtea.iloc[:,[1,2]]
scaler = MinMaxScaler()
data1 = scaler.fit(data).transform(data)
data2 = pd.DataFrame(data1,columns=data.columns)

#常用命令
#深度复制
data1 = data.copy(deep = True)
#合并
data_cleaner = [data1, data_val]
#构造新特征
dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch']
#筛选
dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0
#提取
data.iloc([0],[0])#position
data.loc([0],['a'])#label
data.ix(2)#all
data.ix(:,'a')
data.ix(1,'a')
#字符分割
dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
#分箱（qcut等量，cut等距，数字为份数）
dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)
dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)
#转成列表
data1_x_dummy = data1_dummy.columns.tolist()
#返回序列标签和值
for index, value in enumerate(seq):
    print(index, value)