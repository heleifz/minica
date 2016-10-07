# -*- coding: utf-8 -*-

import idx2numpy
import numpy as np
import os
import sys
import json

import minica.net as net
import minica.tensor as tensor
import minica.solvers.sgd_solver as sgd_solver

# 载入 net
print "Loading MNIST dataset..."
n = net.Net(json.load(open("example_mnist.json")))
train_image = idx2numpy.convert_from_file(open("/Users/helei/Desktop/train-images-idx3-ubyte")).astype(float) / 255.0
train_label = idx2numpy.convert_from_file(open("/Users/helei/Desktop/train-labels-idx1-ubyte")).astype(int)
test_image = idx2numpy.convert_from_file(open("/Users/helei/Desktop/t10k-images-idx3-ubyte")).astype(float) / 255.0
test_label = idx2numpy.convert_from_file(open("/Users/helei/Desktop/t10k-labels-idx1-ubyte")).astype(int)
print "Done"

print "Loading neural net config..."
conf = json.loads(open("example_solver_config.json").read())['params']
solver = sgd_solver.SGDSolver(conf)
print "Done"

print "Shuffling training data.."
shuffled_index = np.random.permutation(train_image.shape[0])
train_image = train_image[shuffled_index]
train_label = train_label[shuffled_index]
print "Done"

def train_source(batch_size):
    for j in xrange(0, train_image.shape[0], batch_size):
        real_batch_size = batch_size if j + batch_size <= train_image.shape[0] else (train_image.shape[0] - j)
        sample = train_image[j:j+real_batch_size].reshape(real_batch_size, train_image.shape[1], train_image.shape[2])
        label = train_label[j:j+real_batch_size].reshape(real_batch_size,1)
        yield {'x' : tensor.Tensor(sample), 'label' : tensor.Tensor(label)}

def test_source(batch_size):
    for j in xrange(0, test_image.shape[0], batch_size):
        real_batch_size = batch_size if j + batch_size <= test_image.shape[0] else (test_image.shape[0] - j)
        sample = test_image[j:j+real_batch_size].reshape(real_batch_size, test_image.shape[1], test_image.shape[2])
        label = test_label[j:j+real_batch_size].reshape(real_batch_size,1)
        yield {'x' : tensor.Tensor(sample), 'label' : tensor.Tensor(label)}

# 开始优化
print "Training neural net..."
solver.solve(train_source, test_source, n)
print "Done"
