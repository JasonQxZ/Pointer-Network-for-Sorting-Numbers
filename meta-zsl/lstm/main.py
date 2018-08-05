#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikai Wang
"""
import random

import numpy as np
import torch
from torch import nn, optim

from model import net_LSTM
from util import (OmniglotGenerator, get_ms, progress_bar, clip_grads,
                  progress_clean, save_checkpoint)


class model_lstm(object):

    def __init__(self):
        self.seed = 1000
        self.init_seed(self.seed)
        self.report_interval = 200
        self.checkpoint_interval = 10000
        self.checkpoint_path = './'
        self.lstm_layers = 1
        self.num_batches = 100000
        self.rmsprop_lr = 1e-3
        self.rmsprop_momentum = 0.9
        self.rmsprop_alpha = 0.95
        self.nb_class = 5
        self.nb_samples_per_class = 10
        self.batch_size = 16
        self.input_size = 20*20
        self.lstm_output_size = 20*20
        self.data_folder = '..//data/Omniglot/images_background_small1'
        self.test_data_folder = '..//data/Omniglot/images_evaluation'
        self.net = net_LSTM(num_inputs = self.input_size,
                        num_hidden = self.lstm_output_size,
                        num_outputs = self.nb_class,
                        num_layers = self.lstm_layers)
        if torch.cuda.is_available():
            self.net = self.net.cuda()
        self.dataloader = OmniglotGenerator(data_folder=self.data_folder,
                                            batch_size=self.batch_size,
                                            nb_samples=self.nb_class,
                                            nb_samples_per_class=self.nb_samples_per_class, 
                                            max_rotation=1., 
                                            max_shift=1.,
                                            max_iter = self.num_batches)
        self.dataloader4test = OmniglotGenerator(data_folder=self.test_data_folder,
                                            batch_size=self.batch_size,
                                            nb_samples=self.nb_class,
                                            nb_samples_per_class=self.nb_samples_per_class, 
                                            max_rotation=1., 
                                            max_shift=1.,
                                            max_iter = self.num_batches)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.RMSprop(self.net.parameters(),
                                        momentum=self.rmsprop_momentum,
                                        alpha=self.rmsprop_alpha,
                                        lr=self.rmsprop_lr)
    
    
    def init_seed(self,seed=None):
        """Seed the RNGs for predicatability/reproduction purposes."""
        if seed is None:
            seed = int(get_ms() // 1000)
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    def train_model(self):
        # losses = []
        start_ms = get_ms()
        while True:
            try:
                (batch_num), (x, y) = self.dataloader.next()
            except StopIteration:
                break
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            # loss = self.train_batch(x, y)
            # losses += [loss]
            self.train_batch(x, y)
            # Update the progress bar
            # progress_bar(batch_num, self.report_interval, loss)
            progress_bar(batch_num, self.report_interval)

            # Report
            if batch_num % self.report_interval == 0:
                # mean_loss = np.array(losses[-self.report_interval:]).mean()
                mean_time = int(((get_ms() - start_ms) / self.report_interval) / self.batch_size)
                progress_clean()
                _, (x4test, y4test) = self.dataloader4test.next()
                if torch.cuda.is_available():
                    x4test = x4test.cuda()
                    y4test = y4test.cuda()
                self.net.eval()
                loss, accuracies = self.evaluate(x4test, y4test)
                self.net.train()
                # print("Batch %d Loss: %.6f  Time: %d ms/sequence  "%(batch_num,mean_loss,mean_time))
                print("Batch %d   Time: %d ms/sequence  "%(batch_num,mean_time))
                print("Loss on test set:{}".format(loss))
                print("------Accuracy on test set:------")
                print("1-instance: %.6f 2-instance: %.6f 5-instance: %.6f 10-instance: %.6f"%
                (accuracies[0],accuracies[1],accuracies[4],accuracies[9]))
                start_ms = get_ms()

            # # Checkpoint
            # if (self.checkpoint_interval != 0) and (batch_num % self.checkpoint_interval == 0):
            #     save_checkpoint(net=self.net, name='lstm', seed=self.seed, checkpoint_path=self.checkpoint_path, batch_num=batch_num, losses=losses)


    def train_batch(self,X, Y):
        """Trains a single batch."""
        self.optimizer.zero_grad()
        self.net.init_sequence(self.batch_size)
        Y_out = torch.zeros((self.nb_class * self.nb_samples_per_class,self.batch_size,self.nb_class), dtype=torch.float32)
        if torch.cuda.is_available():
            Y_out = Y_out.cuda()
        Y_out = Y_out.scatter_(dim=2,index=Y.unsqueeze(2),value=1)
        y_out = torch.zeros((self.nb_class * self.nb_samples_per_class,self.batch_size,self.nb_class), dtype=torch.float32)
        if torch.cuda.is_available():
            y_out = y_out.cuda()
        for i in range(self.nb_class * self.nb_samples_per_class):
            y_out[i] = self.net(X[i])
        loss = self.criterion(y_out, Y_out)
        loss.backward()
        clip_grads(self.net)
        self.optimizer.step()
        # return loss.item()


    def evaluate(self,X, Y):
        """Evaluate a single batch (without training)."""
        self.net.init_sequence(self.batch_size)
        Y_out = torch.zeros(( self.nb_class * self.nb_samples_per_class,self.batch_size,self.nb_class), dtype=torch.float32)
        if torch.cuda.is_available():
            Y_out = Y_out.cuda()
        Y_out = Y_out.scatter_(dim=2,index=Y.unsqueeze(2),value=1)
        y_out = torch.zeros((self.nb_class * self.nb_samples_per_class,self.batch_size,self.nb_class), dtype=torch.float32)
        if torch.cuda.is_available():
            y_out = y_out.cuda()
        for i in range(self.nb_class * self.nb_samples_per_class):
            y_out[i] = self.net(X[i])
        y_out_indexed = torch.zeros((self.nb_class * self.nb_samples_per_class,self.batch_size), dtype=torch.long)
        if torch.cuda.is_available():
            y_out_indexed = y_out_indexed.cuda()
        for i in range(len(y_out_indexed)):
            for j in range(len(y_out_indexed[0])):
                y_out_indexed[i,j] = torch.argmax(y_out[i,j])
        loss = self.criterion(y_out, Y_out)
        return loss.item(), self.compute_accuracy(y=Y,output=y_out_indexed)


    def compute_accuracy(self,y, output):
        correct = [0] * (self.nb_class * self.nb_samples_per_class)
        total = [0] * (self.nb_class * self.nb_samples_per_class)
        if torch.cuda.is_available():
            y = y.cpu().numpy()
            output = output.cpu().numpy()
        else:
            y = y.numpy()
            output = output.numpy()
        # if args.label_type == 'one_hot':
        # y_decode = one_hot_decode(y)
        # output_decode = one_hot_decode(output)
        # elif args.label_type == 'five_hot':
        #     y_decode = five_hot_decode(y)
        #     output_decode = five_hot_decode(output)
        for i in range(self.batch_size):
            y_i = y[:,i]
            output_i = output[:,i]
            class_count = {}
            for j in range(self.nb_class * self.nb_samples_per_class):
                if y_i[j] not in class_count:
                    class_count[y_i[j]] = 0
                class_count[y_i[j]] += 1
                total[class_count[y_i[j]]] += 1
                if y_i[j] == output_i[j]:
                    correct[class_count[y_i[j]]] += 1
        return [float(correct[i]) / total[i] if total[i] > 0. else 0. for i in range(1, 11)]



def main():
    model = model_lstm()
    model.train_model()


if __name__ == '__main__':
    main()
