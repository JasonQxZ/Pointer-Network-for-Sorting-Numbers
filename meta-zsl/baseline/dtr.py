#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 14:18:36 2018

@author: Hendry
"""
from read_usaa import *
from selfTuning import *
from sklearn.tree import DecisionTreeRegressor
from read_awa import *
from read_usaa import *
xTr,yTr,zTr,xTe,yTe,zTe,prototypes = read_usaa_sl(k=3)
dtr = DecisionTreeRegressor()
dtr.fit(xTr,yTr)
yhat = dtr.predict(xTe)
yhat = np.squeeze(yhat)
zTe = np.squeeze(zTe)
unseen = np.unique(zTe)
prototypes_unseen = prototypes[unseen]
protoself = selfTuning(yhat,prototypes_unseen,50)
dist0 = cdist(yhat,prototypes_unseen)
dist1 = cdist(yhat,protoself)
zPred0 = unseen[np.argmin(dist0,axis=1)]
zPred = unseen[np.argmin(dist1,axis=1)]
acc0 = np.zeros([len(unseen)])
acc1 = np.zeros([len(unseen)])
for i in range(len(unseen)):
	idx = zTe==unseen[i]
	acc0[i] = sum(zTe[idx]==zPred0[idx])/sum(idx)
	acc1[i] = sum(zTe[idx]==zPred[idx])/sum(idx)
	
print('acc0',np.mean(acc0))
print('acc',np.mean(acc1))
	
protoself = selfTuning(yhat,prototypes,100)
dist0 = cdist(yhat,prototypes)
dist1 = cdist(yhat,protoself)
zPred0 = np.argmin(dist0,axis=1)
zPred = np.argmin(dist1,axis=1)


gacc0 = sum(zTe==zPred0)/len(zTe)
gacc1 = sum(zTe==zPred)/len(zTe)

print('G acc0',np.mean(gacc0))
print('G acc',np.mean(gacc1))