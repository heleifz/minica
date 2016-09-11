# -*- coding: utf-8 -*-
"""
设计 Layer 层的接口，最重要的问题之一是谁来管理输入/输出 tensors

由于 numpy 对 C-continuous 的支持较好，也就是说，first dim 的变化最慢
所以整个框架也采用 c-continuous 的格式，输入数据也是 *行向量*

"""

import layer
import logging
import numpy as np
import tensor

# logger = logging.Logger(__name__)

class FullLayer(object):
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
        # 兼容 mini-batch 的数据
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
        reshaped_input = np.reshape(prev_data, (size_of_first_dim, self.input_size))
        reshaped_diff = np.reshape(prev_diff, (size_of_first_dim, self.input_size))

        np.dot(next_diff, self.W.mutable_data().T, reshaped_diff)
        # 计算该层参数的梯度
        np.dot(reshaped_input.T / float(size_of_first_dim),
               next_diff,
               self.W.mutable_diff())

    def mutable_params(self):
        return [self.W, self.b]
