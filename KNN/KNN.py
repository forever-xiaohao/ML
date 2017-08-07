#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/7 21:01
# @Author  : Forever
# @Site    : 
# @File    : KNN.py
# @Software: PyCharm Community Edition

from numpy import *
import operator


## 创建数据集
def createDateSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

## K-近邻算法伪代码步骤：
# （1）计算已知类别数据集中的点与当前点之间的距离
# （2）按照距离递增次序排序
# （3）选取与当前点距离最小的K个点
# （4）确定前K个点所在类别的出现频率
# （5）返回前K个点出现频率最高的类别作为当前点的预测分类
def classify0(inx, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # 得到数据集的行数（即数据集的个数）
    diffMat = tile(inx, (dataSetSize, 1)) - dataSet    # tile(A, B)函数是将A复制B次，如B=（3， 1）表示将A复制3行1列
    sqDiffMat = diffMat**2  #   求平方
    sqDistances = sqDiffMat.sum(axis=1)   # 按行方向求平方和
    distances = sqDistances**0.5    # 开平方
    sortedDisIndicies = distances.argsort()    # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
    classCount = {}
    for i in range(k):  # 选择距离最小的k个点
        voteIlabel = labels[sortedDisIndicies[i]]    # 取出按从小到大排序，最小数据所对应的标签值
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1    # get() 函数返回指定键的值，如果值不在字典中返回默认值
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)    # 排序
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()   #按行读取文件
    numberOfLines = len(arrayOLines)   # 得到文件行数
    returnMat = zeros((numberOfLines, 3))    # 创建对应行和列的0矩阵
    classLabelVector = []
    index = 0
    for line in arrayOLines:       # 遍历数据
        line = line.strip()    #  strip() 方法用于移除字符串头尾指定的字符（默认为空格）
        listFromLine = line.split('\t')    # 分割数据
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))    #listFromLine[-1]：得到listFromLine数组的最后一个元素，即为label
        index += 1
    return returnMat, classLabelVector