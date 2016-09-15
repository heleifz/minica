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
        # 校验的结果
        self.validate_result_name = config['validate_result_tensor']
        
        self.lr = float(config['learning_rate'])

    def valdiate(self, net, validation_source):
        """
        在校验集上校验模型
        """
        total = 0
        result = 0.0
        print "validating model..."
        for batch in validation_source(self.batch_size):
            current = net.forward(batch)
            total += batch.shape[0]
            result += current[self.validate_result_name]
        print "validation result: ", result / total

    def update(self, net, batch, current_iter):
        """
        更新模型
        """
        result = net.forward(batch, 'train')
        net.backward(result, 'train')
        for p in net.mutable_params():
            # 更新模型参数
            p.apply_diff(self.lr)

    def solve(training_source, validation_source, net):
        """
        训练网络
        training_source 和 validation_source 接受一个 batch_size 参数
        返回一个数据迭代器
        """
        # 先随机 shuffle 一次
        training_set.shuffle()
        current_iter = 1
        for epoch in range(self.epoch):
            for batch in training_source(self.batch_size):
                self.update(net, batch, current_iter, current_iter)
                if current_iter % self.validate_freq == 0:
                    self.validate(net, validation_source)
                if current_iter % self.snapshot_freq == 0:
                    print "save snapshot...."
                current_iter += 1
                print "====== Finish epoch %d ======" % epoch
