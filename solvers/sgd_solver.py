# -*- coding: utf-8 -*-
"""
"""

class SGDSolver(object):
    """
    随机梯度下降
    """

    def __init__(self, config):
        """
        解析参数
        """

        self.epoch = int(config['epoch'])
        self.snapshot_freq = int(config['snapshot_freq'])
        self.validate_freq = int(config['validate_freq'])
        self.epoch = int(config['epoch'])
        self.batch_size = int(config['batch_size'])
        self.validate_result_tensor = config['validate_result_tensor']

        self.lr = float(config['learning_rate'])

    def valdiate(self, net, validation_set):
        """
        在校验集上校验模型
        """
        pass

    def update(self, net, batch, current_iter):
        """
        更新模型
        """
        pass

    def solve(training_set, validation_set, net):
        """
        训练网络
        """
        # 先随机 shuffle 一次
        training_set.shuffle()
        current_iter = 1
        for epoch in range(self.epoch):
            batch = training_set.read(self.batch_size)
            if batch is None:
                training_set.reset()
            else:
                self.update(net, batch, current_iter, current_iter)
                if current_iter % self.validate_freq == 0:
                    self.validate(net, validation_set)
                if current_iter % self.snapshot_freq == 0:
                    # TODO : 实现模型的序列化和反序列化
                    pass
                current_iter += 1
            print "====== Finish epoch %d ======" % epoch
