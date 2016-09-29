# -*- coding: utf-8 -*-
"""
设计 Layer 层的接口，最重要的问题之一是谁来管理输入/输出 tensors

由于 numpy 对 C-continuous 的支持较好，也就是说，first dim 的变化最慢
所以整个框架也采用 c-continuous 的格式，输入数据也是 *行向量*

"""

import logging
import numpy as np
import minica.tensor as tensor

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
        self.output_size = int(params['output_size'])
        # TODO: 其它初始化策略
        # 不初始化 W, 根据第一次遇到的数据来确定 input_size
        self.W = tensor.Tensor()
        data = np.random.random((1, 1))
        self.b = tensor.Tensor()
        self.b.set_data(data)

    def init_weights(self, input_size, output_size):
        var = 2.0 / float(input_size)
        W = np.random.normal(0, np.sqrt(var), (input_size, output_size))
        self.input_size = input_size
        self.W.set_data(W)

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
        reshaped_input = np.reshape(prev_data, (size_of_first_dim, -1))
        if self.W.mutable_data() is None:
            # 根据输入数据惰性初始化 input_size 以及 W 的尺寸
            self.init_weights(reshaped_input.shape[1], self.output_size)
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
