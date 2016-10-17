# -*- coding: utf-8 -*-
"""
"""

import logging
import numpy as np
import scipy.ndimage
import minica.tensor as tensor
import minica.optimize.pooling_func as pooling_func

# logger = logging.Logger(__name__)

class PoolingLayer(object):
    """
    下采样层
    """

    def __init__(self, params):
        """
        初始化 layer
        """
        self.type = params['type']
        if self.type != "max" and self.type != "mean":
            Exception ("Illegal pooling type:" + self.type)
        self.window_size = tuple(params['window_size'])
        self.stride = tuple(params['stride'])
        self.pooling_buf = None

    def __getstate__(self):
        return {
            "type" : self.type,
            "window_size" : self.window_size,
            "stride" : self.stride
        }
    def __setstate__(self, state):
        self.type = state['type']
        self.window_size = state['window_size']
        self.stride = state['stride']
        self.pooling_buf = None

    def forward(self, prev_tensors, next_tensors):
        """
        前向传播操作
        """
        if len(prev_tensors) != 1:
            raise Exception("Number of input must be 1 for PoolingLayer.")

        prev_data = prev_tensors[0].mutable_data()
        if len(prev_data.shape) == 2:
            prev_data = prev_data.reshape(1, 1, prev_data.shape[0], prev_data[1])
        elif len(prev_data.shape) == 3:
            prev_data = prev_data.reshape((prev_data.shape[0], 1,
                                           prev_data.shape[1], prev_data.shape[2]))
        elif len(prev_data.shape) == 4:
            # do nothing
            pass
        else:
            raise Exception ("""Input for ConvLayer must have shape
                                (n, channel, height, width)
                                or (n, height, width)""")

        h_last = prev_data.shape[2] - self.window_size[0] + 1
        w_last = prev_data.shape[3] - self.window_size[1] + 1

        # 计算输出层的实际尺寸
        output_height = np.arange(0, h_last, self.stride[0]).size
        output_width = np.arange(0, w_last, self.stride[1]).size

        result = np.zeros((prev_data.shape[0],
                           prev_data.shape[1],
                           output_height,
                           output_width), dtype='float32')

        if self.pooling_buf is None or \
           self.pooling_buf.shape[0] != prev_data.shape[0] or \
           self.pooling_buf.shape[1] != prev_data.shape[1] or \
           self.pooling_buf.shape[2] != output_height or \
           self.pooling_buf.shape[3] != output_width:
            self.pooling_buf = np.zeros((prev_data.shape[0],
                                         prev_data.shape[1],
                                         output_height,
                                         output_width), dtype='int32')

        if self.type == "max":
            pooling_func.max_pooling_batch(prev_data, result, self.pooling_buf,
                        self.window_size[0], self.window_size[1],
                        self.stride[0], self.stride[1])

        elif self.type == "mean":
            pooling_func.mean_pooling_batch(prev_data, result, self.pooling_buf,
                        self.window_size[0], self.window_size[1],
                        self.stride[0], self.stride[1])

        output_tensor = tensor.Tensor(result)
        next_tensors.append(output_tensor)

    def backward(self, prev_tensors, next_tensors):
        """
        反向传播操作
        """
        next_diff = next_tensors[0].mutable_diff()
        prev_diff = prev_tensors[0].mutable_diff()

        # 将 prev_diff reshape 成正确形状
        prev_diff = prev_tensors[0].mutable_diff()
        if len(prev_diff.shape) == 2:
            prev_diff = prev_diff.reshape(1, 1, prev_diff.shape[0], prev_diff[1])
        elif len(prev_diff.shape) == 3:
            prev_diff = prev_diff.reshape((prev_diff.shape[0], 1,
                                           prev_diff.shape[1], prev_diff.shape[2]))

        if self.type == "max":
            # 使用 forward 保存的 index 来赋值
            pooling_func.backprop_for_max_pooling(prev_diff, next_diff, self.pooling_buf)

        elif self.type == "mean":

            # 使用 forward 保存的 index 来赋值
            pooling_func.backprop_for_mean_pooling(prev_diff, next_diff, self.pooling_buf,
                                                   self.window_size[0], self.window_size[1])

    def mutable_params(self):
        """
        本层无参数
        """
        return []
