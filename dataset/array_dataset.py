# -*- coding: utf-8 -*-

import numpy as np
import minicaffe.tensor as tensor

class ArrayDataset(object):

    def __init__(self):
        self.data = np.array([])
        self.label = None
        self.idx = []
        self.pos = 0

    def set_data(self, data, label=None):
        """
        设置数据
        """
        self.data = np.array(data)
        if label is not None:
            label = np.array(label)
            if len(label.shape) > 1:
                label = label.reshape((label.size, ))

        self.label = label
        if label is not None and self.data.shape[0] != self.label.shape[0]:
            raise Exception("Shape of data and label don't match.")
        self.idx = np.arange(self.data.shape[0])

    def read(self, num=1):
        """
        从数据集中读取一条或多条数据
        """
        if self.pos + num > self.data.shape[0]:
            return None
        indices = self.idx[self.pos : self.pos + num]
        self.pos += num
        tensor_data = tensor.Tensor()
        tensor_data.set_data(self.data[indices, ...])
        if self.label is not None:
            tensor_label = tensor.Tensor()
            tensor_label.set_data(self.label[indices, ...])
            return (tensor_data, tensor_label)
        else:
            return (tensor_data, None)

    def reset(self):
        """
        重置读取位置
        """
        self.pos = 0

    def shuffle(self):
        """
        重置读取顺序, 保证下次读取时顺序不同
        """
        np.random.shuffle(self.idx)
