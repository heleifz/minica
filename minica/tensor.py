# -*- coding: utf-8 -*-

import numpy as np
import cPickle

class Tensor(object):
    """
    Tensor 对象是网络中所有数据的抽象
    """

    def __init__(self, data=None):
        """
        使用形状初始化 Tensor
        shape 是一个 tuple
        """
        if data is None:
            self._data = None
            self._diff = None
        else:
            self._data = data
            self._diff = np.zeros_like(self._data)

    def apply_diff(self, multiplier):
        """
        将 diff 从 data 上减去，并清空 diff
        """
        self._data -= multiplier * self._diff
        self._diff.fill(0.0)

    def set_data(self, data):
        if self._data is not None and self._data.shape == data.shape:
            self._data = data
            self._diff.fill(0.0)
        else:
            self._data = data
            self._diff = np.zeros_like(self._data)

    def set_diff(self, diff):
        if diff.shape != self._data.shape:
            raise Exception("cannot set_diff, diff and data does't match.")
        self._diff = diff

    def mutable_data(self):
        """
        返回数据，可以修改
        """
        return self._data

    def mutable_diff(self):
        """
        返回 diff, 可以修改
        """
        return self._diff

    def serialize_to_stream(self, stream):
        """
        序列化到一个流中
        """
        cPicke.dump({"data" : self._data, "diff" : self._diff}, stream)

    def deserialize_from_stream(self, stream):
        """
        从流中反序列化
        """
        result = cPickle.load(stream)
        self._data = result['data']
        self._diff = result['diff']
