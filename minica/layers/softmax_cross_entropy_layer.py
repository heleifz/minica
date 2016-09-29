# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import minica.tensor as tensor

class SoftmaxCrossEntropyLayer(object):
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

        if len(prev_tensors) != 2:
            raise Exception("Number of input must be 2 for SoftmaxCrossEntropyLayer.")
        # 兼容 mini-batch 的数据
        prev_predictions = prev_tensors[0].mutable_data()
        prev_labels = prev_tensors[1].mutable_data()
        if len(prev_predictions.shape) == 1:
            raise Exception("Number of dimension must >= 2")
        size_of_first_dim = prev_predictions.shape[0]
        # 变换成行向量
        reshaped_predictions = np.reshape(prev_predictions, (size_of_first_dim, -1))
        reshaped_labels = prev_labels.reshape(-1).astype(int)
        # 每个 prediction 减去最大值
        reshaped_predictions -= reshaped_predictions.max(axis=1, keepdims=True)

        exp_prev = np.exp(reshaped_predictions)
        log_sum_exp = np.log(exp_prev.sum(axis=1, keepdims=True))

        log_softmax = reshaped_predictions - log_sum_exp
        log_probs = log_softmax[np.arange(size_of_first_dim), reshaped_labels]
        self.last_log_softmax = log_softmax
        loss = np.array([-np.sum(log_probs) / size_of_first_dim])

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

        reshaped_labels = prev_labels.reshape(-1).astype(int)
        diff = np.exp(self.last_log_softmax)
        diff[np.arange(size_of_first_dim), reshaped_labels] -= 1
        diff *= next_tensors[0].mutable_diff() / float(size_of_first_dim)
        diff.reshape(prev_predictions.shape)
        prev_tensors[0].set_diff(diff)

        # 不传播到 label (diff 设为 0)
        label_diff = prev_tensors[1].mutable_diff()
        label_diff.fill(0)

    def mutable_params(self):
        # 本层无参数
        return []
