# -*- coding: utf-8 -*-
"""
网络层负责读取/检查网络配置,
以及初始化网络

构建网络
* 和 caffe 不同，这里不存在提供数据的网络节点，输入由外部输入网络，运算需要支持 mini-batch
* 网络构建后，首先对所有节点进行拓扑排序, 确定依赖关系
* 所有 type 为 variable 的节点都需要提供输入
* 所有悬空的输出节点都是输出，所有的输出同时贡献梯度

运算：
* 所有 tensor 尺寸在计算时自动确定
* forward 操作需要用 kwarg 提供所有输入 tensor
* forward 的计算顺序根据拓扑排序
* backward 的计算顺序根据拓扑逆序
* 梯度都是针对 tensor 的，运算负责传递梯度
"""

import json
import importlib
import logging
import sys
import cPickle
from collections import defaultdict

import minica.tensor

logger = logging.Logger(__name__)
logger.addHandler(logging.NullHandler())

class GraphNode(object):
    """
    运算网络的节点，可以是运算节点，也可以是 tensor 节点
    """

    def __init__(self, n_type, name, data=None):
        """
        初始化图节点
        """
        self._name = name
        self._n_type = n_type
        self._prev_nodes = []
        self._next_nodes = []
        self._data = data

    def set_data(self, data):
        self._data = data

    def data(self):
        return self._data

    def node_type(self):
        """
        节点类型：operation, tensor
        """
        return self._n_type

    def node_name(self):
        """
        节点名称
        """
        return self._name

    def mutable_prev_node_names(self):
        """
        前驱节点
        """
        return self._prev_nodes

    def mutable_next_node_names(self):
        """
        后继节点
        """
        return self._next_nodes

def tensor_order_topological(start_node_names, node_table):
    """
    拓扑排序，用于确定运算图的计算顺序
    如果有圈则报错
    """

    result = []
    visited = set()
    current_stack = set()
    def dfs(start_node):
        node_name = start_node.node_name()
        node_type = start_node.node_type()
        visited.add(node_name)
        current_stack.add(node_name)
        for next_node_name in start_node.mutable_next_node_names():
            if next_node_name in current_stack:
                raise ValueError('Computation graph cannot have loop.')
            if next_node_name not in visited:
                dfs(node_table[next_node_name])
        current_stack.remove(node_name)
        if node_type == 'layer':
            # 只关心 layer 节点的拓扑顺序
            result.append(node_name)

    for name in start_node_names:
        if name not in visited:
            dfs(node_table[name])

    result.reverse()
    logger.debug("Topological order for net:" + str(result))

    return result

class Net(object):
    """
    神经网络
    """

    def save(self, path):
        cPickle.dump(self, open(path, 'w'), 1)

    @staticmethod
    def load(path):
        return cPickle.load(open(path))

    def __init__(self, config):
        self.rebuild(config)
        self.config = config

    def mutable_params(self):
        """
        返回整个网络的参数 tensor
        """
        return self.params

    def mutable_learning_rate_multiplier(self):
        return self.learning_rate_multiplier

    def __setstate__(self, state):
        self.rebuild(state['config'], state['layers'])
        self.iter = state['iter']
        self.config = state['config']

    def __getstate__(self):
        return {
            "config" : self.config,
            "layers" : self.layer_table,
            "iter" : self.iter
        }

    def rebuild(self, config, layer_table=None):
        """
        输入网络配置，构建网络
        检查网络结构，确定计算顺序
        """
        self.iter = 0
        self.name = config['name'] if 'name' in config else 'default'
        self.description = config['description'] if 'description' in config else ''

        # 保存 graph node 的表： phase : name: GraphNode
        self.node_table = defaultdict(dict)
        # 保存 layer 的表： name(_phase) : Layer
        if layer_table is None:
            self.layer_table = dict()
        else:
            self.layer_table = layer_table

        # 训练和测试用的 input/output 配置, phase : (name, shape)
        self.variables = defaultdict(list)
        self.outputs = defaultdict(list)

        # 各个 phase 的拓扑顺序
        self.order = dict()

        # 整个网络的可优化参数
        self.params = defaultdict(list)
        self.learning_rate_multiplier = defaultdict(list)

        structure = config['structure']
        # 创建所有 graph node 和 layer 对象, 填充 layer_table 和 node_table
        all_phases = ('train', 'test')
        for layer_config in structure:
            layer_type = layer_config['type']
            name = layer_config['name']
            if "propagate_mask_for_input" in layer_config:
                propagate_mask_for_input = layer_config['propagate_mask_for_input']
            else:
                propagate_mask_for_input = None
            phases = all_phases
            if 'phase' in layer_config:
                phases = [layer_config['phase']]
            if layer_type == 'variable':
                # 处理输入节点
                for p in phases:
                    self.variables[p].append(name)
                    if name in self.node_table[p]:
                        raise ValueError("Variable cannot have same name: %s" % name)
                    self.node_table[p][name] = GraphNode('variable', name)
            else:
                # 创建 layer 对象，layer 文件命名规则: 层名 + _layer
                module_name = 'minica.layers.' + layer_type + '_layer'
                class_name = ''.join([part.title()
                             for part in layer_type.split('_')]) + "Layer"
                module = importlib.import_module(module_name)

                param = {}
                if 'param' in layer_config:
                    param = layer_config['param']

                if 'phase' in layer_config:
                    name = name + "|" + layer_config['phase']


                # 如果没有提供初始化好的 layer, 就重新初始化
                if layer_table is None:
                    if name in self.layer_table:
                        raise ValueError("Layers have same name: %s" % name)
                    param['propagate_mask_for_input'] = propagate_mask_for_input
                    layer = module.__dict__[class_name](param)
                    self.layer_table[name] = layer
                # 否则就 load 外部提供的 layer
                else:
                    layer = self.layer_table[name]

                layer_params = layer.mutable_params()

                if "learning_rate_multiplier" not in layer_config:
                    learning_rate_multiplier = [1] * len(layer_params)
                else:
                    learning_rate_multiplier = \
                        layer_config['learning_rate_multiplier']
                    if len(learning_rate_multiplier) != len(layer_params):
                        raise ValueError("Learning rate multiplier and layer params have different length.")

                # 连接前驱和后继节点
                inputs = layer_config['input']
                outputs = layer_config['output']
                for p in phases:
                    self.params[p].extend(layer_params)
                    self.learning_rate_multiplier[p].\
                        extend(learning_rate_multiplier)
                    layer_node = GraphNode('layer', name,
                        {'layer' : layer, 'config' : layer_config})
                    self.node_table[p][name] = layer_node
                    for inp in inputs:
                        if inp in self.node_table[p]:
                            tensor_node = self.node_table[p][inp]
                        else:
                            tensor_node = GraphNode('tensor', inp)
                            self.node_table[p][inp] = tensor_node
                        # tensor -> layer
                        nexts = tensor_node.mutable_next_node_names()
                        nexts.append(name)
                        prevs = layer_node.mutable_prev_node_names()
                        prevs.append(inp)

                    for out in outputs:
                        if out in self.node_table[p]:
                            tensor_node = self.node_table[p][out]
                        else:
                            tensor_node = GraphNode('tensor', out)
                            self.node_table[p][out] = tensor_node
                        # layer -> tensor
                        nexts = layer_node.mutable_next_node_names()
                        nexts.append(out)
                        prevs = tensor_node.mutable_prev_node_names()
                        prevs.append(name)

        for p in all_phases:
            # 找到所有输出 tensor
            phase_node_table = self.node_table[p]
            for node in phase_node_table.values():
                # 如果一个 node 没有后继节点，那么它就是输出节点
                if len(node.mutable_next_node_names()) == 0:
                    self.outputs[p].append(node.node_name())

            # 拓扑排序
            self.order[p] = tensor_order_topological(
                self.variables[p], self.node_table[p]
            )


    def forward(self, input_tensors, phase):
        """
        整个网络的前向传播
        返回 dict: tensor_name : value
        """
        result = dict()
        for inp in self.variables[phase]:
            result[inp] = input_tensors[inp]
        # 按照拓扑图来计算前向传播
        for layer_name in self.order[phase]:
            # 获取当前 layer
            layer_object = self.layer_table[layer_name]
            layer_node = self.node_table[phase][layer_name]

            # 获取输入 tensor
            input_node_names = layer_node.mutable_prev_node_names()
            input_tensors = [result[n] for n in input_node_names]
            output_node_names = layer_node.mutable_next_node_names()
            output_tensors = []
            # 由 layer 的 forward 接口填充 output tensor
            # 每个 layer 可以在内部使用同一个 ndarray 输出到结果中
            # 避免重复的内存分配
            layer_object.forward(input_tensors, output_tensors)
            # 将结果填入 result
            for idx in xrange(len(output_node_names)):
                result[output_node_names[idx]] = output_tensors[idx]
        return result

    def backward(self, forward_result, phase):
        """
        General Back-propagation
        整个网路的反向传播
        输入 forward 的结果
        """
        # 用一个 dict 累加梯度，如果运算网络的一个 tensor 被多个分支使用
        # 那么，相对于这个 tensor 的梯度是累加的
        gradient_accumulator = dict()
        # 将输出 tensor 的 diff 全部设成 1, 作为 backprop 的边界情况
        for output in self.outputs[phase]:
            forward_result[output].mutable_diff().fill(1.0)
            gradient_accumulator[output] = forward_result[output].mutable_diff()

        # 逆拓扑顺序
        for layer_name in self.order[phase][::-1]:

            # 获取当前 layer
            layer_object = self.layer_table[layer_name]
            layer_node = self.node_table[phase][layer_name]

            # 梯度从 output -> input

            # 初始化 layer 依赖的 tensors
            output_node_names = layer_node.mutable_next_node_names()
            output_tensors = []
            for n in output_node_names:
                current_output = forward_result[n]
                current_output.set_diff(gradient_accumulator[n])
                output_tensors.append(current_output)

            input_node_names = layer_node.mutable_prev_node_names()
            input_tensors = []
            for n in input_node_names:
                current_input = forward_result[n]
                current_input.mutable_diff().fill(0.0)
                input_tensors.append(current_input)

            # 调用 layer 的 backward 接口完成计算
            layer_object.backward(input_tensors, output_tensors)

            # 累加梯度
            for idx, n in enumerate(input_node_names):
                if n in gradient_accumulator:
                    gradient_accumulator[n] += input_tensors[idx].mutable_diff()
                else:
                    gradient_accumulator[n] = input_tensors[idx].mutable_diff()
