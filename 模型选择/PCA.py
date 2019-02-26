
# coding: utf-8

# In[8]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

#数据导入、检查空缺值
data = pd.read_csv(r'D:\fan_fault\already_made_file\train_difference_mean.csv', encoding = "gbk")
train = data.iloc[:,:-1]
label = data.iloc[:,-1]

#数据标准化
scaler = MinMaxScaler()
train = scaler.fit(train).transform(train)

x_train,x_test, y_train, y_test = train_test_split(train, label, train_size=0.7,random_state=1)
#steps = [("pca", PCA()),
#        ("las", LassoCV(alphas=np.logspace(-3,0,10), cv=3, normalize=True))]  #把数据处理过程打包在pip中

steps = [("pca", PCA())]  #把数据处理过程打包在pip中
pip = Pipeline(steps)
gsea = GridSearchCV(pip, param_grid={'pca__n_components': np.arange(1,70,2)}, cv=3, n_jobs = -1) #参数选择在（1,370）中每隔10选一个数,共37个数。
gsea.fit(x_train, y_train)
#print (gsea.score(x_test, y_test))
print (gsea.best_params_)

