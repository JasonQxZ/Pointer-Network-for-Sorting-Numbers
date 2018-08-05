#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikai Wang
"""
import os
import random
import time
from copy import deepcopy

import numpy as np
import torch
from scipy.misc import imread, imresize
from scipy.ndimage import rotate, shift


def get_ms():
    """Returns the current time in miliseconds."""
    return time.time() * 1000


def progress_clean():
    """Clean the progress bar."""
    print("\r{}".format(" " * 80), end='\r')


# def progress_bar(batch_num, report_interval, last_loss):
#     """Prints the progress until the next report."""
#     progress = (((batch_num-1) % report_interval) + 1) / report_interval
#     fill = int(progress * 40)
#     print("\r[{}{}]: {} (Loss: {:.4f})".format(
#         "=" * fill, " " * (40 - fill), batch_num, last_loss), end='')

def progress_bar(batch_num, report_interval):
    """Prints the progress until the next report."""
    progress = (((batch_num-1) % report_interval) + 1) / report_interval
    fill = int(progress * 40)
    print("\r[{}{}]: {}".format(
        "=" * fill, " " * (40 - fill), batch_num), end='')


def save_checkpoint(net, name, seed, checkpoint_path, batch_num, losses):
    progress_clean()
    basename = "{}/{}-{}-batch-{}".format(checkpoint_path, name, seed, batch_num)
    model_fname = basename + ".model"
    print("Saving model checkpoint to: '%s'", model_fname)
    torch.save(net.state_dict(), model_fname)



def clip_grads(net):
    """Gradient clipping to the range [10, 10]."""
    parameters = list(filter(lambda p: p.grad is not None, net.parameters()))
    for p in parameters:
        p.grad.data.clamp_(-10, 10)


def get_shuffled_images(paths, labels, nb_samples=None):
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x:x
    images = [(i, os.path.join(path, image)) for i,path in zip(labels,paths) for image in sampler(os.listdir(path)) ]
    random.shuffle(images)
    return images


def time_offset_label(labels_and_images):
    labels, images = zip(*labels_and_images)
    time_offset_labels = (None,) + labels[:-1]
    return zip(images, time_offset_labels)


def load_transform(image_path, angle=0., s=(0,0), size=(20,20)):
    #Load the image
    original = imread(image_path, flatten=True)
    #Rotate the image
    rotated = np.maximum(np.minimum(rotate(original, angle=angle, cval=1.), 1.), 0.)
    #Shift the image
    shifted = shift(rotated, shift=s)
    #Resize the image
    resized = np.asarray(imresize(shifted, size=size), dtype=np.float32) / 255 
    #Invert the image
    inverted = 1. - resized
    max_value = np.max(inverted)
    if max_value > 0:
        inverted /= max_value
    return inverted


class OmniglotGenerator(object):
    """Docstring for OmniglotGenerator"""
    def __init__(self, data_folder, batch_size=1, nb_samples=5, nb_samples_per_class=10, 
                max_rotation=-np.pi/6, max_shift=10, img_size=(20,20), max_iter=None):
        super(OmniglotGenerator, self).__init__()
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.nb_samples = nb_samples
        self.nb_samples_per_class = nb_samples_per_class
        self.max_rotation = max_rotation
        self.max_shift = max_shift
        self.img_size = img_size
        self.max_iter = max_iter
        self.num_iter = 0
        self.character_folders = [os.path.join(self.data_folder, family, character) \
                                  for family in os.listdir(self.data_folder) \
                                  if os.path.isdir(os.path.join(self.data_folder, family)) \
                                  for character in os.listdir(os.path.join(self.data_folder, family))]
        tmp_character_folders = deepcopy(self.character_folders)
        self.character_folders = [folder for folder in tmp_character_folders if not folder.endswith('.DS_Store')]

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self,shuffle=True):
        if (self.max_iter is None) or (self.num_iter < self.max_iter):
            self.num_iter += 1
            return (self.num_iter - 1), self.sample()
        else:
            raise StopIteration

    def sample(self):
        sampled_character_folders = random.sample(self.character_folders, self.nb_samples)
        random.shuffle(sampled_character_folders)
        example_inputs = torch.zeros((self.nb_samples * self.nb_samples_per_class, self.batch_size, np.prod(self.img_size)), dtype=torch.float32)
        example_outputs = torch.zeros((self.nb_samples * self.nb_samples_per_class,self.batch_size), dtype=torch.long)     
        for i in range(self.batch_size):
            labels_and_images = get_shuffled_images(sampled_character_folders, range(self.nb_samples), nb_samples=self.nb_samples_per_class)
            sequence_length = len(labels_and_images)
            labels, image_files = zip(*labels_and_images)
            angles = np.random.uniform(-self.max_rotation, self.max_rotation, size=sequence_length)
            shifts = np.random.uniform(-self.max_shift, self.max_shift, size=sequence_length)
            example_inputs[:,i] = torch.from_numpy(np.asarray([load_transform(filename, angle=angle, s=shift, size=self.img_size).flatten() \
                                            for (filename, angle, shift) in zip(image_files, angles, shifts)], dtype=np.float32))
            example_outputs[:,i] = torch.from_numpy(np.asarray(labels, dtype=np.int32))
        return example_inputs, example_outputs
