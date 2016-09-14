# -*- coding: utf-8 -*-
"""
对于 layer 层，统一采用测试文件来组织 unittest
所有数值相关的测试全部放在 test_loader 完成
每个 layer 的其它类型的测试放在各自的文件中
"""

import unittest
import minicaffe.layers.full_layer as full_layer
import minicaffe.layers.relu_layer as relu_layer
import minicaffe.layers.softmax_layer as softmax_layer
import minicaffe.layers.conv_layer as conv_layer
import minicaffe.layers.pooling_layer as pooling_layer
import minicaffe.util.gradient_checker as gradient_checker
import minicaffe.tensor as tensor
import numpy as np

class TestLoader(unittest.TestCase):

    def gradient_checker_test(self):
        checker = gradient_checker.GradientChecker(1e-2, 1e-3)
        # layer = full_layer.FullLayer({
        #  "output_size" : 3
        # })
        # layer = softmax_layer.SoftmaxLayer(None)
        # layer = relu_layer.ReluLayer(None)
        # layer = conv_layer.ConvLayer({
        #     "filter_num" : 2,
        #     "filter_size" : 3,
        #     "has_bias" : 1
        # })
        layer = pooling_layer.PoolingLayer({
            "type" : "max",
            "window_size" : [4, 5],
            "stride" : [2, 2]
        })
        input_tensors = []
        t = tensor.Tensor()
        t.set_data(np.random.uniform(10, 1000, (9, 3, 8, 7)))
        input_tensors.append(t)
        self.assertTrue(checker.check(layer, input_tensors))
