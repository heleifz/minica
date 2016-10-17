# -*- coding: utf-8 -*-
"""
对于 layer 层，统一采用测试文件来组织 unittest
所有数值相关的测试全部放在 test_loader 完成
每个 layer 的其它类型的测试放在各自的文件中
"""

import unittest
import numpy as np
import os
import glob
import json
import importlib

import minica.util.gradient_checker as gradient_checker
import minica.tensor as tensor

# 获取测试数据路径
test_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0],
                        "testdata", "layer_test")
class LayerGradientTest(unittest.TestCase):

    def gradient_checker_test(self):
        checker = gradient_checker.GradientChecker(1e-2, 1e-2)
        # 遍历测试目录下的所有文件
        pattern = os.path.join(test_dir, '*.json')
        files = glob.glob(pattern)
        for f in files:
            print 'Testing file:', f
            # load test cases
            conf = json.load(open(f))
            layer_type = conf['layer']['type']
            layer_param = conf['layer']['param']
            test_cases = conf['testcases']
            step_size = conf['checker']['step_size']
            threshold = conf['checker']['threshold']
            checker = gradient_checker.GradientChecker(step_size, threshold)
            input_check_mask = None
            param_check_mask = None
            if "input_check_mask" in conf["checker"]:
                input_check_mask = conf["checker"]["input_check_mask"]
            if "param_check_mask" in conf["checker"]:
                param_check_mask = conf["checker"]["param_check_mask"]

            for c in test_cases:
                # initialize layer for every test cases
                module_name = 'minica.layers.' + layer_type + '_layer'
                class_name = ''.join([part.title()
                             for part in layer_type.split('_')]) + "Layer"
                module = importlib.import_module(module_name)
                layer = module.__dict__[class_name](layer_param)
                inputs = []
                for i in c['inputs']:
                    arr = eval(i)
                    inputs.append(tensor.Tensor(arr))
                self.assertTrue(checker.check_layer(layer, inputs,
                                input_check_mask, param_check_mask))


        # layer = full_layer.FullLayer({
        #  "output_size" : 3
        # })
        # layer = softmax_layer.SoftmaxLayer(None)
        # layer = relu_layer.ReluLayer(None)
        # layer = conv_layer.ConvLayer({
        #     "filter_num" : 3,
        #     "filter_size" : 3,
        #     "has_bias" : 1
        # })
        # layer = pooling_layer.PoolingLayer({
        #     "type" : "mean",
        #     "window_size" : [3, 3],
        #     "stride" : [2, 2]
        # })
        # layer = cross_entropy_layer.CrossEntropyLayer(None)
        # layer = softmax_cross_entropy_layer.SoftmaxCrossEntropyLayer(None)
        # layer = mean_squred_error_layer.MeanSquaredErrorLayer(None)
        # input_tensors = []
        # t = tensor.Tensor()
        # t.set_data(np.random.random((3, 3, 5, 5)).astype('float32') * 1)
        # input_tensors.append(t)
        # t = tensor.Tensor()
        # t.set_data(np.random.randint(0, 6, (5, 1)))
        # input_tensors.append(t)
        # self.assertTrue(checker.check(layer, input_tensors))
