# Intro

minica 是用 Python 实现的深度学习库，本人 learning deep learning 的产出。
目前只有 CPU 版本，用 Cython 加速了卷积和 pooling。

# TODO

* 框架
  * 测试机制：
    * perf test (??)
    * net gradient checker
    * data driven test
  * 完善的输入检查

* 功能
  * 支持 concate 层
  * 支持 dropout 层
  * 支持 word2vec 训练（embedding 层，h softmax，negsamp）

* 应用 example：
  1. 分类(MNIST, CIFAR)
  2. word2vec
  3. style transfer
