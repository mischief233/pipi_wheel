# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 17:58:45 2018

@author: jinqiu
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

#数据集名称为dataset
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_ploly = poly.fit_transform(dataset)
data_ploly = pd.DataFrame(X_ploly, columns=poly.get_feature_names())
new_data = data_ploly.ix[:,73:]

#ploly_mean,ploly_var为交叉特征均值方差
ploly_mean = np.mean(new_data)
ploly_var = np.var(ploly_mean)
