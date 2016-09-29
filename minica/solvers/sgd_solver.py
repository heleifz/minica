# -*- coding: utf-8 -*-
"""
基本的 SGD 算法
"""

import numpy as np
import logging

class SGDSolver(object):
    """
    随机梯度下降
    """

    def __init__(self, config):
        """
        解析参数, 参考 caffe 的 solver 参数
        """
        # 保存模型的间隔
        self.snapshot_interval = int(config['snapshot_interval'])
        # 模型路径的前缀
        self.snapshot_prefix = config['snapshot_prefix']
        # 校验结果的间隔
        self.validate_interval = int(config['validate_interval'])
        # 打印运行日志的间隔
        self.print_log_interval = int(config['print_log_interval'])
        # epoch 的个数
        self.epoch = int(config['epoch'])
        # 训练时的 batch size
        self.batch_size = int(config['batch_size'])
        # 正则化策略 (L1, L2)
        self.regularize_policy = config['regularize_policy']
        self.weight_decay = float(config['weight_decay'])
        # 校验时的 batch size
        self.validate_batch_size = int(config['validate_batch_size'])
        # 校验的结果所在的 tensor 名称
        self.validate_result_name = config['validate_result_name']
        self.base_lr = float(config['base_learning_rate'])
        self.momentum = float(config['momentum'])
        # 最大的迭代次数
        self.max_iter = int(config['max_iteration'])
        # 学习率更新策略
        self.lr_policy = config['learning_rate_policy']
        # 学习率更新策略的参数
        self.lr_policy_params = config['learning_rate_policy_params']
        # 上一轮迭代的梯度
        self.last_diff = None

    def get_learning_rate(self, current_iter):
        """
        直接翻译 caffe 的学习率更新的策略
        """
        if self.lr_policy == "fixed":
            return self.base_lr
        elif self.lr_policy == "step":
            current_step = current_iter / int(self.lr_policy_params['step_size'])
            return self.base_lr * (self.lr_policy_params['gamma'] ** current_step)
        elif self.lr_policy == "exp":
            return self.base_lr * (self.lr_policy_params['gamma'] ** current_iter)
        elif self.lr_policy == "inv":
            return self.base_lr * \
                ((1.0 + self.lr_policy_params['gamma'] * current_iter) **
                 (-self.lr_policy_params['power']))
        elif self.lr_policy == "poly":
            return self.base_lr * ((1.0 - float(current_iter) / self.max_iter) **
                self.lr_policy_params['power'])
        else:
            raise Exception("Unknown learning rate policy:" + self.lr_policy)

    def validate(self, net, validation_source):
        """
        在校验集上校验模型
        """
        total = 0
        result = 0.0
        print "validating model..."
        for batch in validation_source(self.validate_batch_size):
            current = net.forward(batch, 'test')
            total += self.validate_batch_size
            result += current[self.validate_result_name].mutable_data()[0] * self.validate_batch_size
        print "validation result: ", result / float(total)

    def update(self, net, batch, current_iter):
        """
        更新模型
        """
        result = net.forward(batch, 'train')
        net.backward(result, 'train')
        print "loss:", result['loss'].mutable_data()
        # 获取当前迭代的学习率
        lr = self.get_learning_rate(current_iter)
        idx = 0
        for mul, p in zip(net.mutable_learning_rate_multiplier()['train'],
                          net.mutable_params()['train']):
            self.regularize(p)
            # 更新模型参数
            if self.momentum > 0 and self.last_diff[idx] is not None:
                actual_diff = lr * mul * p.mutable_diff() + \
                              self.momentum * self.last_diff[idx]
                self.last_diff[idx] = actual_diff.copy()
                p.set_diff(actual_diff)
                p.apply_diff(1.0)
            else:
                if self.momentum > 0:
                    self.last_diff[idx] = lr * mul * p.mutable_diff().copy()
                p.apply_diff(lr * mul)
            idx += 1

    def regularize(self, param):
        """
        对一个参数进行正则化
        """
        if self.weight_decay > 0:
            diff = param.mutable_diff()
            if self.regularize_policy == "L2":
                diff += self.weight_decay * param.mutable_data()
            elif self.regularize_policy == "L1":
                diff += self.weight_decay * np.sign(param.mutable_data())
            else:
                raise Exception("Unknown regularization policy:" + self.regularize_policy)

    def solve(self, training_source, validation_source, net):
        """
        训练网络
        training_source 和 validation_source 接受一个 batch_size 参数
        返回一个数据迭代器
        """
        current_iter = net.iter + 1
        done = False
        # 保存上一轮的梯度
        self.last_diff = [None for i in xrange(len(net.mutable_params()['train']))]
        for epoch in range(self.epoch):
            for batch in training_source(self.batch_size):
                self.update(net, batch, current_iter)
                if current_iter % self.print_log_interval == 0:
                    print "current iter:", current_iter
                if current_iter % self.validate_interval == 0:
                    self.validate(net, validation_source)
                if current_iter % self.snapshot_interval == 0:
                    # TODO : model 序列化
                    print "save snapshot...."
                if current_iter == self.max_iter:
                    done = True
                    break
                current_iter += 1
            if done:
                break
            print "====== Finish epoch %d ======" % epoch
