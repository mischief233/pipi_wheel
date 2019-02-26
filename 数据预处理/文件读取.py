# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#导入库
import numpy as np
import pandas as pd
import glob
import os

#读取多少个文件glob
#字符查找*
#导入为右上斜杠
path = glob.glob(r'C:/Users/jinqiu/Desktop/a/discrete*.csv')
result = []
for i in path:
    #是否有列名
    dataset = pd.read_csv(i, header=None, encoding = "gbk")
    result.append(dataset)

#字符匹配
path = glob.glob(r'C:/Users/jinqiu/Desktop/new/*.csv')
for i in path:
    #分割
    a=i.split('\\')
    hh=a[-1].split('.')
    test = pd.read_csv(i,encoding = "gbk")
    list1 = []
    #判断文件是否存在，生成文件
    rs = os.path.exists('C:/Users/jinqiu/Desktop/remove/'+hh[0]+'.txt')
    if rs ==True:
        file_handler =open('C:/Users/jinqiu/Desktop/remove/'+hh[0]+'.txt',mode='r')
        #按行读取
        contents = file_handler.readlines()
        for name in contents:
            #换行
            name = name.strip('\n')
            #写下列名集合 
            list1.append(name)
            #删除改集合列
    test = test.drop(list1,axis=1)
    file_handler.close()
    test.to_csv(r'C:/Users/jinqiu/Desktop/new2/'+hh[0]+'.csv',index=False)

#修改列名
result[4].columns = ['偏航要求值_'+str(i) for i in range(3)]
result[3].rename(columns={'偏航状态值_5':'偏航状态值_6'}, inplace = True)

#合并，1为列合并
a = pd.concat(result, axis=1)
b = pd.read_csv(r'C:/Users/jinqiu/Desktop/a/discrete_1.csv',encoding = "gbk")
#fames = [data1,data]
#all_data = pd.concat(fames,axis= 1)

#pd.read_csv(r'C:/Users/jinqiu/Desktop/train-csvdata/train_diff_mean.csv' ,skiprows = [0,],header = None,encoding = "gbk")
a.columns = b.columns

#文件导出，不带标签
a.to_csv(r'C:/Users/jinqiu/Desktop/a/discrete.csv',index=False)

