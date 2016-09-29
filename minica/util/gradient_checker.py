# -*- coding: utf-8 -*-
"""
梯度检查模块
"""
import numpy as np

class GradientChecker(object):

    def __init__(self, step_size, compare_threshold):
        """
        step_size: 数值微分的步长
        step_size: 比较 backprop 和数值微分结果的阈值
        """
        self.step_size = step_size
        self.compare_threshold = compare_threshold

    def compute_fake_loss(self, output_tensors):
        # 给 output 层计算一个假的 loss (平方和除2，diff就是元素本身)
        total_loss = 0.0
        for t in output_tensors:
            data = t.mutable_data()
            diff = t.mutable_diff()
            data_count = data.shape[0]
            total_loss += np.sum(data * data)
            np.copyto(diff, data / float(data_count))
        return total_loss / 2.0 / float(data_count)

    def check(self, layer, input_tensors,
              input_check_mask=None, param_check_mask=None):
        """
        梯度检查, 通过返回 True, 否则返回 False
        假定 layer 已经初始化完毕
        """
        # 需要做梯度检查的 tensor
        check_tensors = []

        params = layer.mutable_params()
        for idx, p in enumerate(params):
            if param_check_mask is None or param_check_mask[idx]:
                check_tensors.append(p)

        for idx, inp in enumerate(input_tensors):
            if input_check_mask is None or input_check_mask[idx]:
                check_tensors.append(inp)

        output_tensors = []
        layer.forward(input_tensors, output_tensors)
        fake_loss = self.compute_fake_loss(output_tensors)
        layer.backward(input_tensors, output_tensors)

        # 开始计算数值微分
        for t in check_tensors:
            data = t.mutable_data()
            diff = t.mutable_diff()
            print t
            for i in xrange(data.size):
                data.flat[i] -= self.step_size
                current_out = []
                layer.forward(input_tensors, current_out)
                neg_loss = self.compute_fake_loss(current_out)
                data.flat[i] += 2 * self.step_size
                current_out = []
                layer.forward(input_tensors, current_out)
                pos_loss = self.compute_fake_loss(current_out)
                data.flat[i] -= self.step_size
                numeric_grad = (pos_loss - neg_loss) / 2.0 / self.step_size
                if np.abs(diff.flat[i] - numeric_grad) > self.compare_threshold:
                    return False
        return True
