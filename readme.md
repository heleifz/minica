# Intro

mini caffe 是用 numpy 实现的类似 caffe 的神经网络库

# TODO

* 卷积层实现错了，每个 channel 需要有一个 2d * filter（cython优化？）
* sgd solver，模型序列化，统一 logging 机制
* 测试机制：
  * perf test
  * net gradient checker
  * load test data
* pooling 层 cython 优化
* 支持 dropout 层
* 支持 word2vec 训练（embedding 层，h softmax，negsamp）
