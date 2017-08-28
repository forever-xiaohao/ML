#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/10 21:52
# @Author  : Forever
# @Site    : 
# @File    : testKDTree.py
# @Software: PyCharm Community Edition

from time import clock
from random import random
from KDTree import *


# 产生一个k维随机向量，每维分量值在0~1之间
def random_point(k):
    return [random() for _ in range(k)]


# 产生n个k维随机向量
def random_points(k, n):
    return [random_point(k) for _ in range(n)]


if __name__ == "__main__":
    data = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]  # samples

    kd = KdTree(data)

    ret = find_nearest(kd, [3, 4.5])
    print ret

    N = 400000
    t0 = clock()
    kd2 = KdTree(random_points(3, N))  # 构建包含四十万个3维空间样本点的kd树
    ret2 = find_nearest(kd2, [0.1, 0.5, 0.8])  # 四十万个样本点中寻找离目标最近的点
    t1 = clock()
    print "time: ", t1 - t0, "s"
    print ret2