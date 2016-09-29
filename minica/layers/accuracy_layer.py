# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import minica.tensor as tensor

class AccuracyLayer(object):
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
            raise Exception("Number of input must be 2 for AccuracyLayer.")
        # 兼容 mini-batch 的数据
        prev_predictions = prev_tensors[0].mutable_data()
        prev_labels = prev_tensors[1].mutable_data()
        reshaped_labels = prev_labels.reshape((-1))
        reshaped_predictions = prev_predictions.reshape((-1))

        match_result = np.sum(reshaped_labels == reshaped_predictions)
        accu = match_result / float(prev_predictions.size)

        next_tensor = tensor.Tensor()
        next_tensor.set_data(np.array([accu]))
        next_tensors.append(next_tensor)

    def backward(self, prev_tensors, next_tensors):
        """
        反向传播操作
        """
        # 准确度无法 backprop 传播
        prediction_diff = prev_tensors[0].mutable_diff()
        label_diff = prev_tensors[1].mutable_diff()
        label_diff.fill(0)
        prediction_diff.fill(0)

    def mutable_params(self):
        # 无参数
        return []
