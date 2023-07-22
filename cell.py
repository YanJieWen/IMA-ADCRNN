# -*- coding:utf-8 -*-
"""
Software:PyCharm
File: cell.py
Institution: --- CSU&BUCEA ---, China
E-mail: obitowen@csu.edu.cn
Author：Yanjie Wen
Date：2023年04月21日
My zoom: https://github.com/YanJieWen
"""

import tensorflow as tf
import scipy.sparse as sp
import numpy as np
from tools import fs_logging
from tools.scripts import *

class AE_DCRNNCell(tf.keras.layers.Layer):
    def __init__(self, num_units, input_dim,adj_mx, diffusion_steps, num_nodes, embed_units, output_dim,if_adp=True,
                 if_spatial=True, activation='tanh', log_dir='./tools/cells.log',**kwargs):
        '''
        实现自适应扩展扩散卷积循环单元
        :param num_units: 单元中的隐层单元数
        :param adj_mx: 邻接矩阵
        :param diffusion_steps:扩散k-hops
        :param num_nodes: 空间节点数
        :param adp_units:自适应矩阵中的节点嵌入维度
        :param temp_units:扩展图卷积的分解临时维度
        :param output_dim:循环单元输出的维度
        :param if_adp:bool是否执行自适应，如果不是，则仅考虑双向随机游走
        :param if_spatial:bool是否考虑空间异质性，如果不是，则仅为普通的图卷积无拓展
        :param if_proj:bool是否输出映射层，如果不是，则输出的为隐形特征
        :param activation:GRU中默认的tanh函数
        :param kwargs:
        '''
        super().__init__(**kwargs)
        self.nodevec = None
        self.adp = None
        self._activation = tf.keras.layers.Activation(activation)
        self.num_nodes = num_nodes
        self.num_units = num_units
        self.diffusion_steps = diffusion_steps
        self.embed_units = embed_units
        self.if_spatial = if_spatial
        self.if_adp = if_adp
        self.output_dim = output_dim
        self.state_size = num_units * num_nodes  # 自定义循环单元，必须要有的
        #===============================================================================================================
        #adp参数
        self.activation = tf.keras.activations.get('relu')
        self.transform_node1 = tf.keras.layers.Dense(self.embed_units)
        self.transform_node2 = tf.keras.layers.Dense(self.embed_units)
        # ===============================================================================================================
        self.input_dim = input_dim
        self.rnnact = tf.keras.layers.Activation('sigmoid')
        self.supports = []
        self._logger = fs_logging.Logger(log_dir,'INFO','INFO')
        temp_supports = []
        temp_supports.append(calculate_random_walk_matrix(adj_mx).T)
        temp_supports.append(calculate_random_walk_matrix(adj_mx.T).T)
        for support in temp_supports:
            self.supports.append(self._build_sparse_matrix(support))
        if self.if_adp:
            self.num_supports = 3
        else:
            self.num_supports =2
        #添加Dropout
        # self.training = training
        # self.dropout1 = tf.keras.layers.Dropout(rate=0.1)
        # self.dropout2 = tf.keras.layers.Dropout(rate=0.1)


    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        return tf.cast(tf.sparse.reorder(L), tf.float32)

    def build(self, batch_input_shape):
        self.num_mx = self.num_supports * self.diffusion_steps + 1
        #创建扩展图卷积，参考文献：Bai L, Yao L, Li C, et al. Adaptive graph convolutional recurrent network for traffic forecasting[J].
        # Advances in neural information processing systems, 2020, 33: 17804-17815
        if self.if_adp or self.if_spatial:
            self.nodevec1 = self.add_weight(shape=[self.num_nodes, self.embed_units],
                                name='nodevec1')
            self.nodevec2 = self.add_weight(shape=[self.num_nodes, self.embed_units],
                                name='nodevec2')
        if self.if_spatial:
            self.ru_wg = self.add_weight(name='ru_weight_pool',
                                         shape=[self.embed_units, self.num_mx,self.input_dim+self.num_units, self.num_units * 2])#[d,k,c_in,c_out]
            self.ru_bg = self.add_weight(name='ru_bias_pool', shape=[self.embed_units, self.num_units * 2])#[d,c_out]

            self.can_wg = self.add_weight(name='can_weight_pool', shape=[self.embed_units, self.num_mx,self.input_dim+self.num_units, self.num_units])#[d,k,c_in,c_out]
            self.can_bg = self.add_weight(name='can_bias_pool', shape=[self.embed_units, self.num_units])#[d,k,c_in,c_out]
        # 如果不考虑空间异质性
        else:
            self.row_kernel = (self.input_dim + self.num_units) * self.num_mx
            self.ru_w = self.add_weight(name='ru_weight_or', shape=[self.row_kernel, self.num_units * 2])
            self.ru_b = self.add_weight(name='ru_bias_or', shape=[ self.num_units * 2,])
            self.c_w = self.add_weight(name='can_weight_or', shape=[self.row_kernel, self.num_units])
        super().build(batch_input_shape)

    @staticmethod
    def concat(x, x_):
        return tf.concat([x, tf.expand_dims(x_, axis=0)], axis=0)

    def _diff_conv(self, inputs, state, new_supports,gate):  # inputs->(b,nd0),state->(b,nd1),根据GRU单元的输入
        input = tf.reshape(inputs, [-1, self.num_nodes, self.input_dim])
        state = tf.reshape(state, [-1, self.num_nodes, self.num_units])
        input_and_state = tf.concat([input, state], axis=-1)
        input_size = input_and_state.shape[-1]
        x = input_and_state
        x0 = tf.transpose(x, [1, 2, 0])
        x0 = tf.reshape(x0, [self.num_nodes, -1])  # n,bd
        x = tf.expand_dims(x0, axis=0)
        # 差分图卷积，参考：Li Y, Yu R, Shahabi C, et al. Diffusion convolutional recurrent neural network: Data-driven traffic forecasting[J].
        # arXiv preprint arXiv:1707.01926, 2017.
        for support in new_supports:
            if not isinstance(support, tf.SparseTensor):
                x1 = tf.matmul(support,x0)
            else:
                x1 = tf.sparse.sparse_dense_matmul(support, x0)#support的dtype应该与x0的dtype对应，均为float32
            x = self.concat(x, x1)
            for _ in range(2, self.diffusion_steps + 1):
                if not isinstance(support, tf.SparseTensor):
                    x2 = 2 * tf.matmul(support, x1) - x0
                else:
                    x2 = 2 * tf.sparse.sparse_dense_matmul(support, x1) - x0
                x = self.concat(x, x2)
                x1, x0 = x2, x1
        self._logger.debug('---一共拼接了{}个张量矩阵---'.format(self.num_mx))
        x = tf.reshape(x, shape=[self.num_mx, self.num_nodes, input_size, -1])
        x = tf.transpose(x, perm=[3, 1, 2, 0])#[b,n,d0+d1,3*s+1]
        x = tf.reshape(x, shape=[-1, input_size * self.num_mx])#[bn,(d0+d1)*(3*s+1)]
        
        if self.if_spatial:
            self._logger.debug('---构建的单元考虑时空异质性---')
            x = tf.transpose(tf.reshape(x, shape=[-1, self.num_nodes, input_size ,self.num_mx]),[0,1,3,2])#[b,n,k,d0+d1]
            #门控块1-type
            nodevec_1 = tf.nn.tanh(self.transform_node1(self.nodevec1))
            nodevec_2 = tf.nn.sigmoid(self.transform_node2(self.nodevec2))
            self.nodevec = nodevec_1*nodevec_2


            if gate == 'reset&update':
                weights_ru = tf.einsum('nd,dkio->nkio',self.nodevec,self.ru_wg)#[n,k,i,o]
                bias_ru = tf.einsum('nd,do->no',self.nodevec,self.ru_bg)
                outputs = tf.einsum('bnki,nkio->bno', x, weights_ru) + bias_ru
                # outputs = self.dropout1(outputs,training=self.training)

            elif gate == 'candidate':
                weights_cu = tf.einsum('nd,dkio->nkio',self.nodevec,self.can_wg)#[n,k,i,o]
                bias_cu = tf.einsum('nd,do->no',self.nodevec,self.can_bg)
                outputs = tf.einsum('bnki,nkio->bno', x, weights_cu) + bias_cu#[b,n,d]
                # outputs = self.dropout2(outputs,training=self.training)
            else:
                raise ValueError
        else:
            if gate == 'reset&update':
                outputs = tf.matmul(x, self.ru_w)
                outputs = tf.nn.bias_add(outputs, self.ru_b)
            elif gate == 'candidate':
                outputs = tf.matmul(x, self.c_w)
                outputs = tf.nn.bias_add(outputs, self.c_b)
            else:
                raise ValueError
        return tf.reshape(outputs, [-1, self.num_nodes * outputs.shape[-1]])
    def call(self, inputs, states):  # (b,nd)
        if self.if_adp:
            self._logger.debug('--构建基于自适应无向图--')
            if self.adp == None:
                _adp = self.activation(tf.matmul(self.nodevec1,tf.transpose(self.nodevec2,[1,0])))
                self.adp = tf.nn.softmax(_adp,axis=-1)#self要提前定义在__init__中
            new_supports = self.supports+[self.adp]
        else:
            new_supports = self.supports
        h_prev = states[0]#标准记忆单元输入，调用RNN会创建一个初始随机张量
        value = self.rnnact(self._diff_conv(inputs, h_prev, new_supports,gate='reset&update'))
        value = tf.reshape(value, (-1, self.num_nodes, self.num_units * 2))
        r, u = tf.split(value=value, num_or_size_splits=2, axis=-1)
        r = tf.reshape(r, (-1, self.num_nodes * self.num_units))
        u = tf.reshape(u, (-1, self.num_nodes * self.num_units))
        c = self._diff_conv(inputs, r * h_prev,new_supports, gate='candidate')
        c = self._activation(c)
        output = new_state = u * h_prev + (1 - u) * c
        return output, [new_state] # ->(b,nd),(b,nd)，记忆单元输出

# 测试代码
# def main():
#     adj_mx = np.ones((10, 10))
#     aedrcnn = AE_DCRNNCell(num_units=64, input_dim=2,adj_mx=adj_mx, diffusion_steps=2, num_nodes=adj_mx.shape[0],
#                            embed_units=10, output_dim=1)
#     Dcgru_layer = tf.keras.layers.RNN(aedrcnn , return_sequences=True, return_state=True)
#     inputs = tf.keras.Input(shape=(12, 10))
#     outputs, state = Dcgru_layer(inputs)
#     model = tf.keras.Model(inputs, outputs)
#     print(inputs.shape,outputs.shape,state.shape)
#     model.summary()

# if __name__ == '__main__':
#     main()