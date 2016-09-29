# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import minica.tensor as tensor

class ArgmaxLayer(object):
    """
    网络层接口
    """

    def __init__(self, params):
        """
        初始化 layer
        """
        # nothing
        pass

    def forward(self, prev_tensors, next_tensors):
        """
        前向传播操作
        """
        if len(prev_tensors) != 1:
            raise Exception("Number of input must be 1 for ArgmaxLayer.")
        prev_data = prev_tensors[0].mutable_data()
        if len(prev_data.shape) == 1:
            prev_data = prev_data.reshape((1, -1))
        size_of_first_dim = prev_data.shape[0]
        reshaped = np.reshape(prev_data, (size_of_first_dim, -1))
        result = np.argmax(reshaped, axis=1)
        next_tensor = tensor.Tensor()
        next_tensor.set_data(result)
        next_tensors.append(next_tensor)

    def backward(self, prev_tensors, next_tensors):
        """
        反向传播操作
        """
        # 无法反向传播
        prev_diff = prev_tensors[0].mutable_diff()
        prev_diff.fill(0.0)

    def mutable_params(self):
        # 无参数
        return []
