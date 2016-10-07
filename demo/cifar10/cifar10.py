# -*- coding: utf-8 -*-

import numpy as np
import os
import sys
import json
import cPickle

import minica.net as net
import minica.tensor as tensor
import minica.solvers.sgd_solver as sgd_solver

if len(sys.argv) != 2:
    print "Usage: python cifar10.py <data_dir>"
    exit(1)

data_dir = sys.argv[1]

# 载入 net
n = net.Net(json.load(open("example_cifar.json")))

print "Loading CIFAR10 dataset..."
# 载入数据集
train_image_batches = []
train_label_batches = []
for i in range(1, 6):
    full_name = open(os.path.join(data_dir, 'data_batch_' + str(i)))
    loaded = cPickle.load(full_name)
    total = loaded['data'].shape[0]
    train_image_batches.append(loaded['data'].reshape((total, 3, 32, 32)).astype('float32') / 255.0)
    train_label_batches.append(np.array(loaded['labels'], dtype='int32').reshape(total, 1))

train_image = np.vstack(train_image_batches)
train_label = np.vstack(train_label_batches)

full_name = open(os.path.join(data_dir, 'test_batch'))
loaded = cPickle.load(full_name)
total = loaded['data'].shape[0]
test_image = loaded['data'].reshape((total, 3, 32, 32)).astype('float32') / 255.0
test_label = np.array(loaded['labels'], dtype='int32').reshape(total, 1)
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
        sample = train_image[j:j+real_batch_size].reshape(real_batch_size, train_image.shape[1], train_image.shape[2], train_image.shape[3])
        label = train_label[j:j+real_batch_size].reshape(real_batch_size,1)
        yield {'x' : tensor.Tensor(sample), 'label' : tensor.Tensor(label)}

def test_source(batch_size):
    for j in xrange(0, test_image.shape[0], batch_size):
        real_batch_size = batch_size if j + batch_size <= test_image.shape[0] else (test_image.shape[0] - j)
        sample = test_image[j:j+real_batch_size].reshape(real_batch_size, test_image.shape[1], test_image.shape[2], test_image.shape[3])
        label = test_label[j:j+real_batch_size].reshape(real_batch_size,1)
        yield {'x' : tensor.Tensor(sample), 'label' : tensor.Tensor(label)}

# 开始优化
print "Training neural net..."
solver.solve(train_source, test_source, n)
print "Done"
