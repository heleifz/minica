# -*- coding: utf-8 -*-
import unittest
import minica.layers.full_layer as full_layer
import numpy as np
import minica.tensor as tensor

class FullLayerTest(unittest.TestCase):

    def setUp(self):
        self.param1 = {
            "input_size" : 4,
            "output_size" : 2
        }
        self.layer = full_layer.FullLayer(self.param1)

    def forward_test(self):
        input_test = np.array([1,2,3,4,5,6,7,8], dtype='float').reshape((2,2,2))
        input_tensor = tensor.Tensor()
        input_tensor.set_data(input_test)
        out = []
        self.layer.forward([input_tensor], out)
        self.layer.backward([input_tensor], out)

    def backward_test(self):
        pass
