# -*- coding: utf-8 -*-
"""
"""

import logging
import numpy as np
import minicaffe.tensor as tensor

# logger = logging.Logger(__name__)

class PoolingLayer(object):
    """
    下采样层
    """

    def __init__(self, params):
        """
        初始化 layer
        """
        self.type = params['type']
        self.window_size = tuple(params['window_size'])
        self.stride = tuple(params['stride'])

    def forward(self, prev_tensors, next_tensors):
        """
        前向传播操作
        """
        if len(prev_tensors) != 1:
            raise Exception("Number of input must be 1 for PoolingLayer.")

        prev_data = prev_tensors[0].mutable_data()
        if len(prev_data.shape) == 2:
            prev_data = prev_data.reshape(1, 1, prev_data.shape[0], prev_data[1])
        elif len(prev_data.shape) == 3:
            prev_data = prev_data.reshape((prev_data.shape[0], 1,
                                           prev_data.shape[1], prev_data.shape[2]))
        elif len(prev_data.shape) == 4:
            # do nothing
            pass
        else:
            raise Exception ("Input for ConvLayer must have shape (n, channel, height, width) or (n, height, width)")

        # TODO maximum filter
        # TODO mean filter


        output_tensor = tensor.Tensor()
        output_tensor.set_data(output_data)
        next_tensors.append(output_tensor)

    def backward(self, prev_tensors, next_tensors):
        """
        反向传播操作
        """

        # 计算反向传播梯度

    def mutable_params(self):
        """
        本层无参数
        """
        return []
