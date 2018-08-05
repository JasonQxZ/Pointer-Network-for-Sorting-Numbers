#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 11:19:28 2018

@author: Hendry
"""
from scipy.spatial.distance import cdist
import numpy as np
def selfTuning(x,proto,k=10):
	dist = cdist(x,proto)
	idx = np.argsort(dist,axis = 0)
	idx = idx[:k,:]
	self_proto = np.zeros_like(proto)
	for i in range(len(self_proto)):
		self_proto[i]=np.mean(x[idx[:,i]],axis=0)
	return self_proto
	