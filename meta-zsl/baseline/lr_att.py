#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 12:59:04 2018

@author: Hendry
"""


import tensorflow as tf
import numpy as np
import random
import datetime
from read_usaa import *
from scipy.spatial.distance import cdist
from selfTuning import *
class linearRegression(object):
	def __init__(self,dimX,dimY,lossFunction=lambda x,y: 
		tf.where(tf.greater(x, y), (x-y), (y-x))  ,regFunction=tf.nn.l2_loss,reg_lambda=10):
		self.x = tf.placeholder(tf.float32, [None, dimX], name="input_x")
		self.y = tf.placeholder(tf.float32, [None, dimY], name="input_y")
		self.b = tf.Variable(tf.random_uniform([dimY], -1.0, 1.0))
		self.W = tf.Variable(tf.random_uniform([dimX, dimY], -1.0, 1.0))
		self.yHat =tf.nn.xw_plus_b(self.x, self.W, self.b)
		losses = lossFunction(self.y,self.yHat)
		self.loss = tf.reduce_mean(losses) + reg_lambda * (regFunction(self.b)+regFunction(self.W))
	

try:
	tf.app.flags.DEFINE_float('learning_rate', 0.001,"learning_rate")
	tf.app.flags.DEFINE_integer('training_epochs', 50,"training_epochs")

	tf.app.flags.DEFINE_float('reg_lambda', 0.01,"reg_lambda")
	tf.app.flags.DEFINE_integer('batch_size', 10,"batch_size")
except:
	pass
unseen = np.array([0,1,3,6])

xTr,yTr,zTr,xTe,yTe,zTe,prototypes = read_usaa_att()

FLAGS = tf.app.flags.FLAGS
def preprocessing(x0,x1):
	mean_x = np.mean(x0)
	std_x = np.std(x0)
	f0 = lambda x: (x-mean_x)/std_x
	return f0(x0),f0(x1)

xTr,xTe = preprocessing(xTr,xTe)
trIdx = [i for i in range(len(xTr))]
step_of_epoch = int(len(xTr)/FLAGS.batch_size)
with tf.Graph().as_default():
	sess = tf.Session()
	with sess.as_default():
		lr = linearRegression(14000,69)
		optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
		grads_and_vars = optimizer.compute_gradients(lr.loss)
		train_op = optimizer.apply_gradients(grads_and_vars)
		sess.run(tf.global_variables_initializer())
		for epoch in range(FLAGS.training_epochs):
			random.shuffle(trIdx)
			for step in range(step_of_epoch):
				_, loss = sess.run([train_op,lr.loss], feed_dict={lr.x: xTr[step*FLAGS.batch_size:(step+1)*FLAGS.batch_size],\
				 lr.y: yTr[step*FLAGS.batch_size:(step+1)*FLAGS.batch_size]})
			time_str = datetime.datetime.now().isoformat()
			print("{}: epoch {}, loss {:g}".format(time_str, epoch, loss))
	
		yhat = sess.run(lr.yHat, feed_dict={lr.x: xTe,lr.y: yTe})
prototypes_unseen = prototypes[unseen]
protoself = selfTuning(yhat,prototypes_unseen,100)
dist0 = cdist(yhat,prototypes_unseen)
dist1 = cdist(yhat,protoself)
zPred0 = unseen[np.argmin(dist0,axis=1)]
zPred = unseen[np.argmin(dist1,axis=1)]
acc0 = np.zeros([4])
acc1 = np.zeros([4])
for i in range(len(unseen)):
	idx = zTe==unseen[i]
	acc0[i] = sum(zTe[idx]==zPred0[idx])/sum(idx)
	acc1[i] = sum(zTe[idx]==zPred[idx])/sum(idx)
	
print('acc0',np.mean(acc0))
print('acc',np.mean(acc1))
	
	
	
	
	
	
	