# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 17:50:45 2018

@author: jinqiu
"""
import numpy as np
import pandas as pd

#分箱（qcut等量，cut等距，数字为份数）
dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)
dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)
