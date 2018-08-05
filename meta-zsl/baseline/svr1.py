# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 15:26:11 2018

@author: Hendry Dong
"""


import sklearn.svm
from read_usaa import *
from selfTuning import *
from read_awa import *
import numpy as np
xTr,yTr,zTr,xTe,yTe,zTe,prototypes = read_awa_sem()
model = []
yhat = np.zeros_like(yTe)

for i in range(len(yTr[0])):
    print('dim',i)
    model.append(sklearn.svm.SVR(kernel='linear'))
    try:
        model[-1].fit(xTr,yTr[:,i])
        yhat[:,i] = model[-1].predict(xTe)
    except:
        yhat[:,i] = np.unique(yTr[:,i])[0]
        


unseen = np.unique(zTe)
prototypes_unseen = prototypes[unseen]
protoself = selfTuning(yhat,prototypes_unseen,50)
dist0 = cdist(yhat,prototypes_unseen)
dist1 = cdist(yhat,protoself)
zPred0 = unseen[np.argmin(dist0,axis=1)]
zPred = unseen[np.argmin(dist1,axis=1)]
acc0 = np.zeros([len(unseen)])
acc1 = np.zeros([len(unseen)])
yhat = np.squeeze(yhat)
zTe = np.squeeze(zTe)
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