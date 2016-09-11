# -*- coding: utf-8 -*-
import unittest
import layers.softmax_layer
import numpy as np
import tensor

class SoftmaxLayerTest(unittest.TestCase):

    def setUp(self):
        self.layer = layers.softmax_layer.SoftmaxLayer(None)

    def forward_test(self):
        input_test = np.array([1, 2, 3, 1, 1, 9], dtype='float').reshape((2,3))
        input_tensor = tensor.Tensor()
        input_tensor.set_data(input_test)
        out = []
        self.layer.forward([input_tensor], out)
        print out[0].mutable_data()
        self.layer.backward([input_tensor], out)
        print input_tensor.mutable_diff()

    def backward_test(self):
        pass
