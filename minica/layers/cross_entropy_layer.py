# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import minica.tensor as tensor

class CrossEntropyLayer(object):
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
            raise Exception("Number of input must be 2 for CrossEntropyLayer.")
        # 兼容 mini-batch 的数据
        prev_predictions = prev_tensors[0].mutable_data()
        prev_labels = prev_tensors[1].mutable_data()
        if len(prev_predictions.shape) == 1:
            raise Exception("Number of dimension must >= 2")
        size_of_first_dim = prev_predictions.shape[0]
        # 变换成行向量
        reshaped_predictions = np.reshape(prev_predictions, (size_of_first_dim, -1))
        reshaped_labels = prev_labels.reshape(-1).astype(int)
        # 获取 label 对应列的概率值
        probs = reshaped_predictions[np.arange(size_of_first_dim), reshaped_labels]
        log_probs = np.log(probs)
        loss = np.array([-np.sum(log_probs) / size_of_first_dim])

        next_tensor = tensor.Tensor()
        next_tensor.set_data(loss)
        next_tensors.append(next_tensor)

    def backward(self, prev_tensors, next_tensors):
        """
        反向传播操作
        """
        prev_predictions = prev_tensors[0].mutable_data()
        prediction_diff = prev_tensors[0].mutable_diff()
        prev_labels = prev_tensors[1].mutable_data()
        size_of_first_dim = prediction_diff.shape[0]

        reshaped_diff = np.reshape(prediction_diff, (size_of_first_dim, -1))
        reshaped_diff.fill(0.0)

        reshaped_predictions = np.reshape(prev_predictions, (size_of_first_dim, -1))
        reshaped_labels = prev_labels.reshape(-1).astype(int)
        reshaped_diff[np.arange(size_of_first_dim), reshaped_labels] = \
            -1.0 / reshaped_predictions[np.arange(size_of_first_dim), reshaped_labels] * \
            next_tensors[0].mutable_diff() / size_of_first_dim

        # 不传播到 label (diff 设为 0)
        label_diff = prev_tensors[1].mutable_diff()
        label_diff.fill(0)

    def mutable_params(self):
        # 无参数
        return []
