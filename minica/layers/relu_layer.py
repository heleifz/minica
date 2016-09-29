# -*- coding: utf-8 -*-
"""
"""
import numpy as np
import minica.tensor as tensor

class ReluLayer(object):
    """
    网络层接口
    """

    def __init__(self, params):
        """
        初始化 layer
        """
        # do nothing
        pass

    def forward(self, prev_tensors, next_tensors):
        """
        前向传播操作
        """
        if len(prev_tensors) != 1:
            raise Exception("Number of input must be 1 for FullLayer.")
        prev_data = prev_tensors[0].mutable_data()
        if len(prev_data.shape) == 1:
            raise Exception("Number of dimension must >= 2")
        next_data = prev_data.copy()
        next_data[next_data < 0] = 0
        next_tensor = tensor.Tensor()
        next_tensor.set_data(next_data)
        next_tensors.append(next_tensor)

    def backward(self, prev_tensors, next_tensors):
        """
        反向传播操作
        """
        next_diff = next_tensors[0].mutable_diff()
        prev_diff = prev_tensors[0].mutable_diff()
        prev_data = prev_tensors[0].mutable_data()
        mask = prev_data < 0
        not_mask = np.logical_not(mask)
        prev_diff[mask] = 0
        prev_diff[not_mask] = next_diff[not_mask]

    def mutable_params(self):
        # 该层无参数
        return []
