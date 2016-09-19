# -*- coding: utf-8 -*-

import idx2numpy
import numpy as np
import os
import sys
import json

sys.path.insert(0, "/Users/helei/project/dl")
import minicaffe.net as net
import minicaffe.tensor as tensor
import minicaffe.solvers.sgd_solver as sgd_solver

# 载入 net
n = net.Net(open("example_mnist.json").read())
train_image = idx2numpy.convert_from_file(open("/Users/helei/Desktop/train-images-idx3-ubyte")).astype(float) / 255.0
train_label = idx2numpy.convert_from_file(open("/Users/helei/Desktop/train-labels-idx1-ubyte")).astype(int)
test_image = idx2numpy.convert_from_file(open("/Users/helei/Desktop/t10k-images-idx3-ubyte")).astype(float) / 255.0
test_label = idx2numpy.convert_from_file(open("/Users/helei/Desktop/t10k-labels-idx1-ubyte")).astype(int)

conf = json.loads(open("/Users/helei/project/dl/minicaffe/example_solver_config.json").read())['params']
solver = sgd_solver.SGDSolver(conf)

print train_image.shape
print train_label.shape
print test_image.shape
print test_label.shape

shuffled_index = np.random.permutation(train_image.shape[0])
train_image = train_image[shuffled_index]
train_label = train_label[shuffled_index]

def train_source(batch_size):
    for j in xrange(0, train_image.shape[0], batch_size):
        sample = train_image[j:j+batch_size].reshape(batch_size, train_image.shape[1], train_image.shape[2])
        label = train_label[j:j+batch_size].reshape(batch_size,1)
        yield {'x' : tensor.Tensor(sample), 'label' : tensor.Tensor(label)}

def test_source(batch_size):
    for j in xrange(0, test_image.shape[0], batch_size):
        sample = test_image[j:j+batch_size].reshape(batch_size, test_image.shape[1], test_image.shape[2])
        label = test_label[j:j+batch_size].reshape(batch_size,1)
        yield {'x' : tensor.Tensor(sample), 'label' : tensor.Tensor(label)}

# 开始优化
solver.solve(train_source, test_source, n)

# for i in xrange(epoch_num):
#     for j in xrange(0, train_image.shape[0], batch_size):
#         iter_num += 1
#         sample = train_image[j:j+batch_size].reshape(batch_size, train_image.shape[1], train_image.shape[2])
#         label = train_label[j:j+batch_size].reshape(batch_size,1)
#         # 梯度下降
#         r = n.forward({'x' : tensor.Tensor(sample), 'label' : tensor.Tensor(label)}, 'train')
#         n.backward(r, 'train')
#         lr = 0.02 - (0.02 - 0.001) * iter_num / total_iter
#
#         for p in n.mutable_params()['train']:
#             p.apply_diff(lr)
#
#         if iter_num % 10 == 0:
#             print "-------- iter %d -----------" % iter_num
#             print "learning rate:", lr
#             print "current loss:", r['loss'].mutable_data()
#
#         if iter_num % 500 == 0:
#             r = n.forward({'x' : tensor.Tensor(test_image), 'label' : tensor.Tensor(test_label)}, 'test')
#             print "accuracy: ", r['accu_out'].mutable_data()
#
#
# # 梯度下降
# r = n.forward({'x' : tensor.Tensor(test_image), 'label' : tensor.Tensor(test_label)}, 'test')
# print "accuracy: ", r['accu_out'].mutable_data()
