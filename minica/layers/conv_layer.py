# -*- coding: utf-8 -*-
"""
卷积层
"""

import logging
import numpy as np
import minica.tensor as tensor
import minica.optimize.conv_func

# logger = logging.Logger(__name__)

class ConvLayer(object):
    """
    网络层接口
    """
    def __init__(self, params):
        """
        初始化 layer
        """
        # logger.info("Initializing ConvLayer.")
        # 滤波器形状
        self.filter_size = int(params['filter_size'])
        # 滤波器个数
        self.filter_num = int(params['filter_num'])
        # 是否有 bias term
        self.has_bias = int(params['has_bias'])

        self.filters = tensor.Tensor()
        # self.filters.set_data(data)
        if self.has_bias:
            data = np.random.random((1, 1))
            self.b = tensor.Tensor()
            self.b.set_data(data)

    def init_weights(self, channel_num, height, width):
        """
        初始化 filter 参数
        """
        var = 2.0 / ((self.filter_size ** 2) * channel_num * 100)
        data = np.random.normal(0, np.sqrt(var),
                                 (self.filter_num, channel_num,
                                  self.filter_size, self.filter_size))
        self.filters.set_data(data)

        # 计算卷积时需要用到的 buffer（用于 padding 和矩阵乘法）
        self.forward_buf = np.zeros((channel_num * self.filter_size * self.filter_size,
                                     (height - self.filter_size + 1) * (width - self.filter_size + 1)))
        self.backward_pad_buf = np.zeros((self.filter_num, height + self.filter_size - 1, width + self.filter_size - 1))
        self.backward_conv_buf1 = np.zeros(((self.filter_size ** 2) * self.filter_num, height * width))
        self.backward_conv_buf2 = np.zeros(((height - self.filter_size + 1) * (width - self.filter_size + 1),
                                             (self.filter_size ** 2) * channel_num))


    def forward(self, prev_tensors, next_tensors):
        """
        前向传播操作
        输入必须是
        (n, channel, height, width)
        或者
        (n, height, width)
        """
        if len(prev_tensors) != 1:
            raise Exception("Number of input must be 1 for ConvLayer.")

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
            raise Exception ("Input for ConvLayer must have shape (n, channel, height, width) or (n, height, width)")

        if self.filters.mutable_data() is None:
            # 根据首次输入的数据的 channel 数来初始化卷积层
            self.init_weights(prev_data.shape[1], prev_data.shape[2], prev_data.shape[3])
        result = np.zeros((prev_data.shape[0], self.filter_num,
                           prev_data.shape[2] - self.filter_size + 1,
                           prev_data.shape[3] - self.filter_size + 1))
        conv_func.conv_batch(prev_data, result,
                             self.filters.mutable_data(), self.forward_buf)
        if self.has_bias:
            result += self.b.mutable_data()

        output_tensor = tensor.Tensor()
        output_tensor.set_data(result)
        next_tensors.append(output_tensor)

    def backward(self, prev_tensors, next_tensors):
        """
        卷积层 backprop
        """
        next_diff = next_tensors[0].mutable_diff()

        # 将 prev_diff reshape 成正确形状
        prev_diff = prev_tensors[0].mutable_diff()
        if len(prev_diff.shape) == 2:
            prev_diff = prev_diff.reshape(1, 1, prev_diff.shape[0], prev_diff[1])
        elif len(prev_diff.shape) == 3:
            prev_diff = prev_diff.reshape((prev_diff.shape[0], 1,
                                           prev_diff.shape[1], prev_diff.shape[2]))

        prev_data = prev_tensors[0].mutable_data()
        if len(prev_data.shape) == 2:
            prev_data = prev_data.reshape(1, 1, prev_data.shape[0], prev_data[1])
        elif len(prev_data.shape) == 3:
            prev_data = prev_data.reshape((prev_data.shape[0], 1,
                                           prev_data.shape[1], prev_data.shape[2]))
        filter_data = self.filters.mutable_data()

        conv_func.backward_for_conv_batch(prev_diff, next_diff, filter_data,
                                          self.backward_pad_buf, self.backward_conv_buf1)
        conv_func.backward_kernel_for_conv_batch(prev_data, next_diff,
                                                 self.filters.mutable_diff(),
                                                 self.backward_conv_buf2)
        # 计算 bias term 的梯度
        if self.has_bias:
            b_diff = self.b.mutable_diff()
            b_diff[0] = next_diff.sum()

    def mutable_params(self):
        if self.has_bias:
            return [self.filters, self.b]
        else:
            return [self.filters]
