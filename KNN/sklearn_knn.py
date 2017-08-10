#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/8 22:53
# @Author  : Forever
# @Site    : 
# @File    : sklearn_knn.py
# @Software: PyCharm Community Edition
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import sklearn

if __name__ == "__main__":
    # 查看iris数据集
    iris = load_iris()
    print "iris数据集：", iris.data
    print "iris标签：", iris.target

    # 分割训练集合测试集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, train_size=0.7, random_state=1)

    # 构建模型
    knn = KNeighborsClassifier(n_neighbors=3)
    # 训练数据集
    knn.fit(x_train, y_train)
    # 预测
    predict = knn.predict(x_test)
    print predict
    print iris.target_names[predict]
    print knn.predict_proba(x_test)
    print knn.score(x_test, y_test)