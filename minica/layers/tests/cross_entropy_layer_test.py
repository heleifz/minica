# -*- coding: utf-8 -*-
import unittest
import minica.layers.cross_entropy_layer as cross_entropy_layer
import numpy as np
import minica.tensor as tensor

class CrossEntropyLayerTest(unittest.TestCase):

    def setUp(self):
        self.layer = cross_entropy_layer.CrossEntropyLayer(None)

    def forward_test(self):
        input_test = np.array([0.1, 0.2, 0.7, 0.5, 0.2, 0.3], dtype='float').reshape((2,3))
        input2_test = np.array([2, 0], dtype='float').reshape((2,1))
        input_tensor = tensor.Tensor()
        input_tensor.set_data(input_test)
        input_tensor2 = tensor.Tensor()
        input_tensor2.set_data(input2_test)
        out = []
        self.layer.forward([input_tensor, input_tensor2], out)
        print out[0].mutable_data()
        self.layer.backward([input_tensor, input_tensor2], out)
        print input_tensor.mutable_diff()

    def backward_test(self):
        pass
