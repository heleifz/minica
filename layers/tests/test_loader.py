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
import minicaffe.util.gradient_checker as gradient_checker
import minicaffe.tensor as tensor
import numpy as np

class TestLoader(unittest.TestCase):

    def gradient_checker_test(self):
        checker = gradient_checker.GradientChecker(1e-2, 1e-3)
        # layer = full_layer.FullLayer({
        #  "input_size" : 5,
        #  "output_size" : 3
        # })
        # layer = softmax_layer.SoftmaxLayer(None)
        # layer = relu_layer.ReluLayer(None)
        input_tensors = []
        t = tensor.Tensor()
        t.set_data(np.array([[1, -2, 3, 0.5, 1], [4, 7, 9, 0.1, -8]]))
        input_tensors.append(t)
        self.assertTrue(checker.check(layer, input_tensors))
