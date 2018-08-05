#!/usr/bin/env pzthon3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 17:36:20 2018

@author: Hendrz
"""

import numpy as np
from spyder.utils.iofuncs import load_dictionary

import h5py

def read_awa(path = './awa_input.mat'):
	data = h5py.File(path)
	xTr = np.array(data['xTr']).T
	xTe = np.array(data['xTe']).T
	zTr = np.squeeze(np.array(data['yTr']).astype('int'))-1
	zTe = np.squeeze(np.array(data['yTe']).astype('int'))-1

	prototypes = np.array(data['att']).T
	yTr = np.squeeze(prototypes[zTr])
	yTe = np.squeeze(prototypes[zTe])
	
	return xTr,yTr,zTr,xTe,yTe,zTe,prototypes
	
def read_awa_bin(path = './awa_input.mat'):
	data = h5py.File(path)
	xTr = np.array(data['xTr']).T
	xTe = np.array(data['xTe']).T
	zTr = np.squeeze(np.array(data['yTr']).astype('int'))-1
	zTe = np.squeeze(np.array(data['yTe']).astype('int'))-1
	dataproto = h5py.File('./awa_attr.mat')
	prototypes = np.array(dataproto['attr']).T
	yTr = np.squeeze(prototypes[zTr])
	yTe =np.squeeze( prototypes[zTe])
	
	return xTr,yTr,zTr,xTe,yTe,zTe,prototypes
def read_awa_sem(path = './awa_input.mat'):
	data = h5py.File(path)
	xTr = np.array(data['xTr']).T
	xTe = np.array(data['xTe']).T
	zTr = np.squeeze(np.array(data['yTr']).astype('int'))-1
	zTe = np.squeeze(np.array(data['yTe']).astype('int'))-1
	dataproto = h5py.File('./awa_proto.mat')
	prototypes = np.array(dataproto['attr']).T
	yTr = np.squeeze(prototypes[zTr])
	yTe =np.squeeze( prototypes[zTe])
	
	return xTr,yTr,zTr,xTe,yTe,zTe,prototypes


