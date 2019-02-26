# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 17:04:01 2018

@author: jinqiu
"""

import pandas as pd
import numpy as np

newtea = pd.read_excel('C:/Users/jinqiu/Desktop/茶叶.xlsx', sheetname='Sheet1')
newtea.info()
newtea.notnull().sum(axis=0)/newtea.shape[0]
#newtea = tea.drop('商品产地',axis=1)
#促销
teapromote = newtea['促销']
a=teapromote.shape[0]

for i in range(0,a):
     if (teapromote[i]!='无'):
         teapromote[i]='有'
         print(teapromote[i])
         
#退货
teaquit = newtea['退货']
for i in range(0,a):   
    if (teaquit[i]=='\ue604支持7天无理由退货'):
        teaquit[i] = '是'
    else:
        teaquit[i] = '否'


tea_brand = newtea['品牌']
area = newtea['商品产地']
brandcounts = tea_brand.value_counts()
brandcounts2 = dict(brandcounts)
otherbrands = []
highbrands=[]
for key,value in brandcounts2.items():
    if value<10:
        otherbrands.append(key)
    elif value >=10:
        highbrands.append(key)
for i in otherbrands:
    for j in range(1403):
        if (i==tea_brand[j]):
            #tea_brand1 = tea_brand.loc[j,'品牌']
            tea_brand[j]='其他品牌'
        else:
            tea_brand[j]='知名品牌'
for i in range(1403):
    if ('其他品牌'!=tea_brand[i]):
        tea_brand[i]='知名品牌'
 
tea_brand[tea_brand.isnull()] = tea_brand.dropna().mode().values
area[area.isnull()] = area.dropna().mode().values

for i in range(1403):
    if ('中国大陆'==area[i]):
        area[i]='无产地'
    else:
        area[i]='有产地' 
    #newtea['品牌'].str.replace(i,'其他品牌')
    #tea_brand.str.replace(i,'其他品牌')
newteacopy = newtea.copy()

'''from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="most_frequent")
X = imputer.transform(tea_brand)
Imputer.fit(tea_brand)
'''
'''newtea['品牌'].value_counts()
tea_brand = newtea['品牌']
Promotions = newtea['促销']
Returns = newtea['退货']'''
'''from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
tea_brand1 = encoder.fit_transform(tea_brand)
Promotions1 = encoder.fit_transform(Promotions)
Returns1 = encoder.fit_transform(Returns)
tea_brand2 = pd.DataFrame(tea_brand1,columns='品牌')'''
newtea['单价']=newtea['价格']/newtea['产品毛重']
tea_brand2=tea_brand.copy()
newtea['品牌']=tea_brand
newtea['商品产地'] = area
newtea['促销'] = teapromote
newtea['退货'] = teaquit

from sklearn.preprocessing import MinMaxScaler
data = newtea.iloc[:,[1,2,3,4,9]]
scaler = MinMaxScaler()
data1 = scaler.fit(data).transform(data)
data2 = pd.DataFrame(data1,columns=data.columns)
newtea1= newtea.copy()
newtea.drop(newtea.columns[5], axis=1,inplace=True)   # Note: zero indexed
#newtea['价格'] = data2
newtea = pd.get_dummies(newtea,columns=['品牌','促销','退货'])
newtea3=pd.concat([newtea,data2],axis=1)
newtea3.to_excel('C:/Users/jinqiu/Desktop/茶叶3.xlsx')


        
    


        
        
