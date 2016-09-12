# -*- coding: utf-8 -*-
"""
卷积层
"""

import logging
import numpy as np
import scipy.signal
import minicaffe.tensor as tensor

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

        data = np.random.random((self.filter_num,
                                 self.filter_size, self.filter_size))
        self.filters = tensor.Tensor()
        self.filters.set_data(data)
        if self.has_bias:
            data = np.random.random((1, 1))
            self.b = tensor.Tensor()
            self.b.set_data(data)

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

        mean_input = prev_data.mean(axis=1)
        self.last_mean = mean_input

        # 输入已经是 (n, h, w) 了
        # 调用 scipy.ndimage.correlate 做卷积
        # 先分配一个连续的内存空间
        filter_data = self.filters.mutable_data()
        result = []
        # 遍历每张图片
        for idx in xrange(prev_data.shape[0]):
            # 对于每个滤波器
            for f in xrange(self.filter_num):
                result.append(scipy.signal.correlate2d(mean_input[idx],
                                                       filter_data[f],
                                                       mode='valid'))
        result = np.stack(result)
        result = result.reshape(prev_data.shape[0],
                                self.filter_num,
                                prev_data.shape[2] - self.filter_size + 1,
                                prev_data.shape[3] - self.filter_size + 1)
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

        filter_data = self.filters.mutable_data()
        # 计算输入的梯度
        for idx in xrange(next_diff.shape[0]):
            for f in xrange(self.filter_num):
                current_diff = next_diff[idx][f]
                prop_diff = scipy.signal.convolve2d(current_diff,
                                                    filter_data[f],
                                                    mode='full')
                # 将梯度平均分配到输入的各个 channel
                # 利用 broadcasting
                prev_diff[idx] += prop_diff / float(prev_diff.shape[1])

        # 对于每个滤波器
        for f in xrange(self.filter_num):
            # 每张图片的梯度累加
            filter_diff = self.filters.mutable_diff()[f]
            filter_diff.fill(0)
            for idx in xrange(self.last_mean.shape[0]):
                current_diff = next_diff[idx][f]
                filter_diff += scipy.signal.correlate2d(self.last_mean[idx],
                                                        current_diff,
                                                        mode='valid')
        # 计算 bias term 的梯度
        if self.has_bias:
            b_diff = self.b.mutable_diff()
            b_diff[0] = next_diff.sum()

    def mutable_params(self):
        if self.has_bias:
            return [self.filters, self.b]
        else:
            return [self.filters]
