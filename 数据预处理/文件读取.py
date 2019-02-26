# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#导入库
import numpy as np
import pandas as pd
import glob

#读取多少个文件glob
#字符查找*
#导入为右上斜杠
path = glob.glob(r'C:/Users/jinqiu/Desktop/a/discrete*.csv')
result = []
for i in path:
    #是否有列名
    dataset = pd.read_csv(i, header=None, encoding = "gbk")
    result.append(dataset)

#修改列名
result[4].columns = ['偏航要求值_'+str(i) for i in range(3)]
result[3].rename(columns={'偏航状态值_5':'偏航状态值_6'}, inplace = True)

#合并，1为列合并
a = pd.concat(result, axis=1)
b = pd.read_csv(r'C:/Users/jinqiu/Desktop/a/discrete_1.csv',encoding = "gbk")
a.columns = b.columns

#文件导出，不带标签
a.to_csv(r'C:/Users/jinqiu/Desktop/a/discrete.csv',index=False)