# -*- coding: utf-8 -*-
"""
"""

import logging
import numpy as np
import scipy.ndimage
import minica.tensor as tensor
import optimize.pooling_func

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

        self.max_pooling_buf = None

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
                           output_width), dtype=float)

        if self.max_pooling_buf is None or \
           self.max_pooling_buf.shape[0] != prev_data.shape[0] or \
           self.max_pooling_buf.shape[1] != prev_data.shape[1] or \
           self.max_pooling_buf.shape[2] != output_height or \
           self.max_pooling_buf.shape[3] != output_width:
            self.max_pooling_buf = np.zeros((prev_data.shape[0],
                                             prev_data.shape[1],
                                             output_height,
                                             output_width), dtype='int32')

        if self.type == "max":
            pooling_func.max_pooling_batch(prev_data, result, self.max_pooling_buf,
                        self.window_size[0], self.window_size[1],
                        self.stride[0], self.stride[1])

        elif self.type == "mean":

            temp_result = np.zeros(
                (prev_data.shape[2], prev_data.shape[3])
            )
            h_start = np.round((self.window_size[0]) / 2.0 + 0.5)
            w_start = np.round((self.window_size[1]) / 2.0 + 0.5)
            h_last = prev_data.shape[2] - ((prev_data.shape[2] - h_start) % self.stride[0])
            w_last = prev_data.shape[3] - ((prev_data.shape[3] - w_start) % self.stride[1])

            position = position[h_start:h_last:self.stride[0],
                                w_start:w_last:self.stride[1]]
            self.position = position.reshape(-1)
            self.temp_diff = np.zeros((prev_data.shape[2], prev_data.shape[3]))
            for idx in xrange(prev_data.shape[0]):
                for ch in xrange(prev_data.shape[1]):
                    current_channel = prev_data[idx][ch]
                    scipy.ndimage.uniform_filter(current_channel,
                                                 size=self.window_size,
                                                 output=temp_result,
                                                 mode='constant')
                    # 按照 window_size 和 stride 取对应点的均值
                    result[idx][ch] = temp_result[h_start:h_last:self.stride[0],
                                                  w_start:w_last:self.stride[1]]

        output_tensor = tensor.Tensor()
        output_tensor.set_data(result)
        next_tensors.append(output_tensor)

    def backward(self, prev_tensors, next_tensors):
        """
        反向传播操作
        """
        next_diff = next_tensors[0].mutable_diff()
        prev_diff = prev_tensors[0].mutable_diff()

        if self.type == "max":
            # 使用 forward 保存的 index 来赋值
            pooling_func.backprop_for_max_pooling(prev_diff, next_diff, self.max_pooling_buf)

        elif self.type == "mean":
            for idx in xrange(next_diff.shape[0]):
                for ch in xrange(next_diff.shape[1]):
                    # 将梯度填入 0 矩阵的对应下标中，然后用均值滤波来分配梯度
                    self.temp_diff.flat[self.position] = next_diff[idx][ch].reshape(-1)
                    scipy.ndimage.uniform_filter(self.temp_diff,
                                                 size=self.window_size,
                                                 output=prev_diff[idx][ch], mode='constant',
                                                 origin=(-1,0))

    def mutable_params(self):
        """
        本层无参数
        """
        return []
