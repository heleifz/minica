# -*- coding: utf-8 -*-
"""
"""

import logging
import numpy as np
import minicaffe.tensor as tensor

# logger = logging.Logger(__name__)

class ConvLayer(object):
    """
    网络层接口
    """

    def __init__(self, params):
        """
        初始化 layer
        """
        # logger.info("Initializing FullLayer.")
        self.input_size = int(params['input_size'])
        self.output_size = int(params['output_size'])
        # TODO: 其它初始化策略
        data = np.random.random((self.input_size, self.output_size))
        self.W = tensor.Tensor()
        self.W.set_data(data)
        data = np.random.random((1, 1))
        self.b = tensor.Tensor()
        self.b.set_data(data)

    def forward(self, prev_tensors, next_tensors):
        """
        前向传播操作
        """
        if len(prev_tensors) != 1:
            raise Exception("Number of input must be 1 for FullLayer.")

        prev_data = prev_tensors[0].mutable_data()
        if len(prev_data.shape) == 1:
            raise Exception("Number of dimension must >= 2")
        size_of_first_dim = prev_data.shape[0]
        reshaped_input = np.reshape(prev_data, (size_of_first_dim, self.input_size))
        # y = Wx + b
        output_data = np.dot(reshaped_input, self.W.mutable_data()) + \
            self.b.mutable_data()
        output_tensor = tensor.Tensor()
        output_tensor.set_data(output_data)
        next_tensors.append(output_tensor)

    def backward(self, prev_tensors, next_tensors):
        """
        反向传播操作
        """
        next_diff = next_tensors[0].mutable_diff()
        # 计算传递到前级的梯度
        prev_data = prev_tensors[0].mutable_data()
        prev_diff = prev_tensors[0].mutable_diff()
        size_of_first_dim = prev_data.shape[0]
        reshaped_input = np.reshape(prev_data, (size_of_first_dim, -1))
        reshaped_diff = np.reshape(prev_diff, (size_of_first_dim, -1))

        # 计算反向传播梯度
        np.dot(next_diff, self.W.mutable_data().T, reshaped_diff)

        # 计算该层参数的梯度
        np.copyto(self.b.mutable_diff(),
                  next_diff.sum())
        np.dot(reshaped_input.T,
               next_diff,
               self.W.mutable_diff())

    def mutable_params(self):
        return [self.W, self.b]
