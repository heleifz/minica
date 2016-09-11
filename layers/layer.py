# -*- coding: utf-8 -*-
"""
设计 Layer 层的接口，最重要的问题之一是谁来管理输入/输出 tensors

"""

class Layer(object):
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
        前向传播操作, next_tensors 总是一个空数组, 输出的 tensor 往里填
        """
        pass

    def backward(self, prev_tensors, next_tensors):
        """
        反向传播操作, prev_tensors 的 diff 部分可以当作是 0
        """
        pass

    def mutable_params(self):
        """
        caffe 中的 param 既属于 layer，又暴露给 solver

        * 属于 layer 的原因是只有 layer 知道怎么计算它的 diff
        * 暴露给 solver 的原因是只有 solver 才知道怎么用它的 diff

        在概念上更干净的模型里，param 也是该层的输入，只不过在计算中是一个 constant，
        但是这让接口过于抽象，按照一般用户的理解，param 和 layer 是绑定的，而不是分离的
        """
        pass
