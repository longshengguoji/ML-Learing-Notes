#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 23:10:39 2016

@author: zhuwei
"""

import KNN

group,labels =KNN.createDataSet()
print (group)
print (labels)

res = KNN.classify0([0,0.5],group,labels,3)
print(res)

print(group.shape[0])

print(tile([0,0.5],(4,1)))