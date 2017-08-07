#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/7 21:24
# @Author  : Forever
# @Site    : 
# @File    : testKNN.py
# @Software: PyCharm Community Edition

from KNN import createDateSet, classify0, file2matrix
import matplotlib
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
    # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],
    #           15.0*array(datingLabels), 15.0*array(datingLabels))
    #使用列1和2展示数据
    # ax.scatter(dating)
    plt.show()