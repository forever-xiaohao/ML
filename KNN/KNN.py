#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/7 21:01
# @Author  : Forever
# @Site    : 
# @File    : KNN.py
# @Software: PyCharm Community Edition

from numpy import *
import operator
from os import listdir


###################################################################################################
# K-近邻算法实现
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


#################################################################################################
# 示例一、约会网站
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


### 归一化特征值，归一化结果：数据集范围：数据集最小值
def autoNorm(dataSet):
    minVals = dataSet.min(0)     # 参数0使得函数可以从列中选取最小值
    maxvals = dataSet.max(0)    # 同样是从列中选择最大值
    ranges = maxvals - minVals
    normDataSet = zeros(shape(dataSet))    # 创建和dataSet维度大小的0矩阵
    m = dataSet.shape[0]    # 返回数据集的行数
    normDataSet = dataSet - tile(minVals, (m, 1))   # 数据集减去每一列的最小值
    normDataSet = normDataSet/tile(ranges, (m, 1))  # 数据集每个数据除
    return normDataSet, ranges, minVals

#   分裂期针对约会网站的测试代码
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix("./dataset/datingTestSet2.txt")    # 得到数据集和标签
    normMat, ranges, minVals = autoNorm(datingDataMat)  # 归一化数据集
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs: m, :],
                                     datingLabels[numTestVecs: m], 3)
        print "The classfier came back with : %d, the real answer is : %d" % (classifierResult, datingLabels[i])
        if(classifierResult != datingLabels[i]):
            errorCount += 1.0
    print "The total error rate is : %f" % (errorCount / float(numTestVecs))

#   使用算法构建完整的可用系统
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input("percentage of time spent playing vido games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix("./dataset/datingTestSet2.txt")   # 读取数据
    normMat, ranges, minVals = autoNorm(datingDataMat)  # 归一化数据
    inArr = array([ffMiles, percentTats, iceCream])    # 输入的测试数据
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)    # 归一化测试数据
    print "You will probably like this person:", resultList[classifierResult - 1]


#########################################################################################################
# 示例：手写识别系统
#########################################################################################################


# 函数img2vector()将图像转换为向量
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()     # 读取一行数据
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

# 手写数字识别系统的测试代码
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir("./dataset/digits/trainingDigits")   # 得到文件列表
    m = len(trainingFileList)   # 包含文件的个数
    trainingMat = zeros((m, 1024))
    for i in range(m):  # 从文件中解析出分类数字和将图像转换为向量
        fileNameStr = trainingFileList[i]   # 得到文件
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])    # 得到图像类别数字
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector("./dataset/digits/trainingDigits/%s" % fileNameStr)
    testFileList = listdir("./dataset/digits/testDigits")
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector("./dataset/digits/testDigits/%s" % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifler came back with: %d, the real answer is : %d" % (classifierResult, classNumStr)
        if(classifierResult != classNumStr):
            errorCount += 1.0
    print "\nthe total number of errors is : %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))


