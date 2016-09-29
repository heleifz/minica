# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import minica.tensor as tensor

class SoftmaxLayer(object):
    """
    网络层接口
    """

    def __init__(self, params):
        """
        初始化 layer
        """
        pass

    def forward(self, prev_tensors, next_tensors):
        """
        前向传播操作
        """
        if len(prev_tensors) != 1:
            raise Exception("Number of input must be 1 for SoftmaxLayer.")
        # 兼容 mini-batch 的数据
        prev_data = prev_tensors[0].mutable_data()
        if len(prev_data.shape) == 1:
            raise Exception("Number of dimension must >= 2")
        size_of_first_dim = prev_data.shape[0]
        # 变换成行向量
        reshaped_input = np.reshape(prev_data, (size_of_first_dim, -1))
        reshaped_input -= reshaped_input.max(axis=1, keepdims=True)
        exp_prev = np.exp(reshaped_input)
        sum_exp = exp_prev.sum(axis=1, keepdims=True)
        next_data = exp_prev / sum_exp
        next_tensor = tensor.Tensor()
        next_tensor.set_data(next_data)
        next_tensors.append(next_tensor)

    def backward(self, prev_tensors, next_tensors):
        """
        反向传播操作
        """
        prev_diff = prev_tensors[0].mutable_diff()
        size_of_first_dim = prev_diff.shape[0]
        reshaped_diff = prev_diff.reshape(size_of_first_dim, -1)
        next_data = next_tensors[0].mutable_data()
        next_diff = next_tensors[0].mutable_diff()

        # 分别对每个输入传播梯度
        for i in xrange(next_data.shape[0]):
            # 根据 output 计算雅可比矩阵
            # (diag(output, output) - output^T * output)^T * next_grad
            # next_grad(row) * jacobian
            prev_one_row_diff = reshaped_diff[i].reshape((1, -1))
            next_one_row_diff = next_diff[i].reshape((1, -1))
            next_one_row = next_data[i].reshape((1, -1))
            jacobian = np.dot(next_one_row.T, next_one_row)
            diag_matrix = np.diag(next_one_row.reshape(-1))
            jacobian = diag_matrix - jacobian
            np.dot(next_one_row_diff, jacobian, prev_one_row_diff)

    def mutable_params(self):
        # 本层无参数
        return []
