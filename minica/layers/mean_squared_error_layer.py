# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import minica.tensor as tensor

class MeanSquaredErrorLayer(object):
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
        if len(prev_tensors) != 2:
            raise Exception("Number of input must be 2 for MeanSquaredErrorLayer.")
        # 兼容 mini-batch 的数据
        prev_predictions = prev_tensors[0].mutable_data()
        prev_labels = prev_tensors[1].mutable_data()
        if len(prev_predictions.shape) == 1:
            raise Exception("Number of dimension must >= 2")
        size_of_first_dim = prev_predictions.shape[0]
        # 变换成行向量
        reshaped_predictions = prev_predictions.reshape(size_of_first_dim, -1)
        reshaped_labels = prev_labels.reshape(size_of_first_dim, -1)
        # 计算 squared error
        loss  = np.array([np.sum((reshaped_predictions - reshaped_labels) ** 2) 
                / float(size_of_first_dim) / 2])

        next_tensor = tensor.Tensor()
        next_tensor.set_data(loss)
        next_tensors.append(next_tensor)

    def backward(self, prev_tensors, next_tensors):
        """
        反向传播操作
        """
        prev_predictions = prev_tensors[0].mutable_data()
        prev_labels = prev_tensors[1].mutable_data()
        size_of_first_dim = prev_predictions.shape[0]
        # 变换成行向量
        reshaped_predictions = prev_predictions.reshape(size_of_first_dim, -1)
        reshaped_labels = prev_labels.reshape(size_of_first_dim, -1)
        diff = (prev_predictions - prev_labels) * next_tensors[0].mutable_diff() \
                / float(size_of_first_dim)
        prev_tensors[0].set_diff(diff.reshape(prev_tensors[0].mutable_data().shape))
        prev_tensors[1].mutable_diff().fill(0)

    def mutable_params(self):
        # 无参数
        return []
