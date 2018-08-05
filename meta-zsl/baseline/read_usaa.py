#!/usr/bin/env pzthon3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 17:36:20 2018

@author: Hendrz
"""

import numpy as np
from spyder.utils.iofuncs import load_dictionary

import h5py

def read_usaa(path = './input.mat',tr = [3,5,6,8]):
	data = h5py.File(path)
	x0 = np.array(data['Xtrain']).T
	x1 = np.array(data['Xtest']).T
	z0 = np.array(data['train_video_label']).astype('int')
	z1 = np.array(data['test_video_label']).astype('int')
	idx0 =np.zeros([1,len(x0)])
	idx1 = np.zeros([1,len(x1)])
	for i in tr:
		idx0+=(z0==i).astype('int')
	idx0 = np.squeeze(idx0.astype('bool'))
	for i in tr:
		idx1+=(z1==i).astype('int')
	idx1 = np.squeeze(idx1.astype('bool'))
	xTr = np.concatenate((x1[idx1],x0[idx0]))
	zTr = np.concatenate((z1[0,idx1],z0[0,idx0]))-1
	idx0 = idx0==False
	idx1 = idx1==False
	xTe = np.concatenate((x1[idx1],x0[idx0]))
	zTe = np.concatenate((z1[0,idx1],z0[0,idx0]))-1
	
	data_dict = load_dictionary('./prototypes_usaa.spydata')
	prototypes = data_dict[0]['prototypes']
	yTr = prototypes[zTr]
	yTe = prototypes[zTe]
	
	return xTr,yTr,zTr,xTe,yTe,zTe,prototypes
	
def read_usaa_exp(path = './input.mat',tr = [3,5,6,8]):
	data = h5py.File(path)
	x0 = np.array(data['Xtrain']).T
	x1 = np.array(data['Xtest']).T
	z0 = np.array(data['train_video_label']).astype('int')
	z1 = np.array(data['test_video_label']).astype('int')
	idx0 =np.zeros([1,len(x0)])
	idx1 = np.zeros([1,len(x1)])
	for i in tr:
		idx0+=(z0==i).astype('int')
	idx0 = np.squeeze(idx0.astype('bool'))
	for i in tr:
		idx1+=(z1==i).astype('int')
	idx1 = np.squeeze(idx1.astype('bool'))
	xTr = np.zeros([len(tr),len(x0[0])])
	xTr0 = np.concatenate((x1[idx1],x0[idx0]))
	zTr0 = np.concatenate((z1[0,idx1],z0[0,idx0]))-1
	zTr = np.array(tr)-1
	for i in range(len(zTr)):
		xTr[i,:] =np.mean(xTr0[zTr0== zTr[i],:],axis = 0)
	idx0 = idx0==False
	idx1 = idx1==False
	xTe = np.concatenate((x1[idx1],x0[idx0]))
	zTe = np.concatenate((z1[0,idx1],z0[0,idx0]))-1
	
	data_dict = load_dictionary('./prototypes_usaa.spydata')
	prototypes = data_dict[0]['prototypes']
	yTr = prototypes[zTr]
	yTe = prototypes[zTe]
	return xTr,yTr,zTr,xTe,yTe,zTe,prototypes
def read_usaa_att(path = './input.mat',tr = [3,5,6,8]):
	data = h5py.File(path)
	x0 = np.array(data['Xtrain']).T
	x1 = np.array(data['Xtest']).T
	y0 = np.array(data['train_attr']).T
	y1 = np.array(data['test_attr']).T
	z0 = np.array(data['train_video_label']).astype('int')
	z1 = np.array(data['test_video_label']).astype('int')
	idx0 =np.zeros([1,len(x0)])
	idx1 = np.zeros([1,len(x1)])
	for i in tr:
		idx0+=(z0==i).astype('int')
	idx0 = np.squeeze(idx0.astype('bool'))
	for i in tr:
		idx1+=(z1==i).astype('int')
	idx1 = np.squeeze(idx1.astype('bool'))
	xTr = np.concatenate((x1[idx1],x0[idx0]))
	yTr = np.concatenate((y1[idx1],y0[idx0]))
	zTr = np.concatenate((z1[0,idx1],z0[0,idx0]))-1
	idx0 = idx0==False
	idx1 = idx1==False
	xTe = np.concatenate((x1[idx1],x0[idx0]))
	yTe = np.concatenate((y1[idx1],y0[idx0]))
	zTe = np.concatenate((z1[0,idx1],z0[0,idx0]))-1
	prototypes = np.zeros([8,len(yTe[0])])
	for i in range(8):
		prototypes[i,:] = (np.mean(y0[z0[0,:]==i+1],axis=0)+np.mean(y1[z1[0,:]==i+1],axis=0))/2
	
	return xTr,yTr,zTr,xTe,yTe,zTe,prototypes

def read_usaa_att_cont(path = './input.mat',tr = [3,5,6,8]):
	data = h5py.File(path)
	x0 = np.array(data['Xtrain']).T
	x1 = np.array(data['Xtest']).T
	y0 = np.array(data['train_attr']).T
	y1 = np.array(data['test_attr']).T
	z0 = np.array(data['train_video_label']).astype('int')
	z1 = np.array(data['test_video_label']).astype('int')
	idx0 =np.zeros([1,len(x0)])
	idx1 = np.zeros([1,len(x1)])
	prototypes = np.zeros([8,len(y0[0])])
	for i in range(8):
		prototypes[i,:] = (np.mean(y0[z0[0,:]==i+1],axis=0)+np.mean(y1[z1[0,:]==i+1],axis=0))/2
	
	for i in tr:
		idx0+=(z0==i).astype('int')
	idx0 = np.squeeze(idx0.astype('bool'))
	for i in tr:
		idx1+=(z1==i).astype('int')
	idx1 = np.squeeze(idx1.astype('bool'))
	xTr = np.concatenate((x1[idx1],x0[idx0]))
	
	zTr = np.concatenate((z1[0,idx1],z0[0,idx0]))-1
	idx0 = idx0==False
	idx1 = idx1==False
	xTe = np.concatenate((x1[idx1],x0[idx0]))
	
	zTe = np.concatenate((z1[0,idx1],z0[0,idx0]))-1
	yTr = prototypes[zTr]
	yTe = prototypes[zTe]
	return xTr,yTr,zTr,xTe,yTe,zTe,prototypes

def read_usaa_full(path = './input.mat',tr = [3,5,6,8]):
	xTr,yTr1,zTr,xTe,yTe1,zTe,prototypes1 = read_usaa(path = './input.mat',tr = [3,5,6,8])
	xTr,yTr2,zTr,xTe,yTe2,zTe,prototypes2 = read_usaa_att(path = './input.mat',tr = [3,5,6,8])
	yTr = np.concatenate((yTr1,yTr2),axis = 1)
	yTe = np.concatenate((yTe1,yTe2),axis = 1)
	prototypes = np.concatenate((prototypes1,prototypes2),axis = 1)
	return xTr,yTr,zTr,xTe,yTe,zTe,prototypes

def read_usaa_sl_sem(path = './input.mat',k=100):
	
	data = h5py.File(path)
	x0 = np.array(data['Xtrain']).T
	xTe = np.array(data['Xtest']).T
	z0 = np.array(data['train_video_label']).astype('int')-1
	zTe = np.array(data['test_video_label']).astype('int')-1
	idx = np.array([])
	labels = np.unique(z0)
	for i in labels:
		idx = np.concatenate((idx,np.squeeze(np.where(z0==i)[1][:k])))
	idx = idx.astype('int')
	xTr = x0[idx,:]
	zTr = z0[0,idx]
	
	data_dict = load_dictionary('./prototypes_usaa.spydata')
	prototypes = data_dict[0]['prototypes']
	yTr = prototypes[zTr]
	yTe = prototypes[zTe]
	
	return xTr,yTr,zTr,xTe,yTe,zTe,prototypes

def read_usaa_sl(path = './input.mat',k=100):
	data = h5py.File(path)
	x0 = np.array(data['Xtrain']).T
	xTe = np.array(data['Xtest']).T
	z0 = np.array(data['train_video_label']).astype('int')-1
	zTe = np.array(data['test_video_label']).astype('int')-1
	y0 = np.array(data['train_attr']).T
	yTe = np.array(data['test_attr']).T
	idx = np.array([])
	labels = np.unique(z0)
	for i in labels:
		idx = np.concatenate((idx,np.squeeze(np.where(z0==i)[1][:k])))
	idx = idx.astype('int')
	xTr = x0[idx,:]
	zTr = z0[0,idx]
	yTr = y0[idx,:]
	y0 = np.array(data['train_attr']).T
	prototypes = np.zeros([8,len(y0[0])])
	x0 = np.array(data['Xtrain']).T
	x1 = np.array(data['Xtest']).T
	
	y1 = np.array(data['test_attr']).T
	z0 = np.array(data['train_video_label']).astype('int')
	z1 = np.array(data['test_video_label']).astype('int')
	for i in range(8):
		prototypes[i,:] = (np.mean(y0[z0[0,:]==i+1],axis=0)+np.mean(y1[z1[0,:]==i+1],axis=0))/2
	
	
	return xTr,yTr,zTr,xTe,yTe,zTe,prototypes

