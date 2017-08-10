#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/7 21:24
# @Author  : Forever
# @Site    : 
# @File    : testKNN.py
# @Software: PyCharm Community Edition

from KNN import createDateSet, classify0, file2matrix, datingClassTest, classifyPerson,\
img2vector, handwritingClassTest
import matplotlib as mp
import matplotlib.pyplot as plt
from numpy import *


if __name__ == "__main__":
    # 加载默认数据集
    data, label = createDateSet();

    print data
    print
    print label

    # 测试KNN算法
    print classify0([0, 0], data, label, 3)

    ## 应用实例一、使用k-近邻算法改进约会网站的配对效果
    # 加载数据集
    datingDataMat, datingLabels = file2matrix("./dataset/datingTestSet2.txt")

    # 查看数据集
    print "数据集为：\n", datingDataMat
    # 对应标签
    print "对应的标签为：\n", datingLabels

    ## 分析数据：使用Matplotlit创建散点图
    fig = plt.figure(figsize=(10, 9), facecolor='w')
    ax = fig.add_subplot(111)
    # 使用列2和3展示数据
    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],
               15.0*array(datingLabels), 15.0*array(datingLabels))
    #使用列1和2展示数据
    # ax.scatter(dating)
    plt.show()

    # 对归一化之后的数据进行预测，并输出错误率
    datingClassTest()
    # 约会网站预测函数
    classifyPerson()

    #################################################################
    # 示例二、使用K-近邻算法识别手写数字
    testVector = img2vector("./dataset/digits/testDigits/0_13.txt")
    print "数字图像转向量示例：", testVector

    # 测试代码
    handwritingClassTest()