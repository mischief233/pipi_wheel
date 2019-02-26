#预测红色圆点标记的电影（101,20）的类别，K取3

#导入模块 numpy科学计算包，operator运算符模块
import numpy as np
import operator

"""
函数说明：创建数据集

parameter：
    无
Return：
    group -数据集
    labels -标签
"""
def createDataSet():
    #数据集中由四组二维特征
    group = np.array([[1,101],[5,89],[108,5],[115,8]])
    #对应分类标签四组特征
    labels = ['爱情片','爱情片','动作片','动作片']
    return group,labels

"""
函数说明：KNN算法，分类器

parames：
    inX - 用于分类的数据（测试集）
    dataSet - 训练集
    labels - 分类标签
    k - KNN算法参数，选择距离最小的K个点
Returns:
    sortedClassCount[0][0] - 分类结果
 """
   
def classify0(inX,dataSet,labels,k):
    #shape[0]返回dataset的行数
    dataSetSize=dataSet.shape[0]
    #将inX横线复制1次，再沿纵向复制dataSetSize次
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet
    #二维特征相减后平方
    sqDiffMat = diffMat**2
    #sum()所有元素相加，sum(0)列相加，sum(1)行相加
    sqDistances = sqDiffMat.sum(1)
    #开方，计算出距离
    distances = sqDistances**0.5
    #返回distances中元素从小到大排序后的索引值
    sortedDistIndices = distances.argsort()
    #定义一个记录类别次数的字典
    classCount = {}
    for i in range(k):
        #取出前K个元素的类别
        voteIlabel = labels[sortedDistIndices[i]]
        #dict.get(key,default=None),字典的get()返回指定健的值，若不在字典中返回默认值
        #计算类别次数
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    #key=operator.items(1)根据字典的值进行排序
    #key=operator.items(0)根据字典的键进行排序
    #reverse降序排列字典
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    #返回次数最多的类别，即所要分类的类别
    return sortedClassCount[0][0]

if __name__=='__main__':
    #创建数据集
    group,labels = createDataSet()
    #测试集
    test =[101,100]
    #Knn分类
    test_class=classify0(test,group,labels,3)
    #打印分类结果
    print(test_class)