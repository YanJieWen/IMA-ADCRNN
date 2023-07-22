# -*- coding:utf-8 -*-
"""
Software:PyCharm
File: model.py
Institution: --- CSU&BUCEA ---, China
E-mail: obitowen@csu.edu.cn
Author：Yanjie Wen
Date：2023年04月21日
My zoom: https://github.com/YanJieWen
"""

import tensorflow as tf
from cell import *

class Multi_head_inter_attention(tf.keras.layers.Layer):#多头BahdanauAttention机制
    def __init__(self,num_nodes,num_dims,num_heads,input_len,horizen,**kwargs):
        super().__init__(**kwargs)
        self.num_nodes = num_nodes
        self.num_dims = num_dims
        self.num_heads = num_heads
        self.wq = tf.keras.layers.Dense(num_dims)
        self.wk = tf.keras.layers.Dense(num_dims)
        self.wv = tf.keras.layers.Dense(num_dims)
        self.dense1 = tf.keras.layers.Dense(num_dims,activation='linear')
        self.pe_embedding_enc = positional_encoding(input_len,num_dims)#(1,te,d)
        self.pe_embedding_dec = positional_encoding(horizen,num_dims)#(1,td,d)
    def call(self,enc_out,dec_out,t):#[b,1,nd]&[b,t,nd]多头注意力机制
        query = tf.reshape(tf.transpose(tf.reshape(dec_out,(-1,dec_out.shape[1],self.num_nodes,self.num_dims)),perm=[0,2,1,3]),(-1,dec_out.shape[1],self.num_dims))#维度聚合可以节省计算资源[bn,td,d]
        values = keys= tf.reshape(tf.transpose(tf.reshape(enc_out,(-1,enc_out.shape[1],self.num_nodes,self.num_dims)),perm=[0,2,1,3]),(-1,enc_out.shape[1],self.num_dims))#(bn,te,d)
        query+=self.pe_embedding_dec[:,t:t+1,:]
        values+=self.pe_embedding_enc
        keys+=self.pe_embedding_enc
        query = self.wq(query)#(bn,td,d)
        query = tf.concat(tf.split(query,self.num_heads,axis=-1),axis=0)#[bnh,td,d/h]
        keys = self.wk(keys)
        keys = tf.concat(tf.split(keys,self.num_heads,axis=-1),axis=0)#[bnh,te,d/h]
        values  = self.wv(values)
        values = tf.concat(tf.split(values,self.num_heads,axis=-1),axis=0)#[bnh,te,d/h]
        att_m = tf.nn.softmax(tf.matmul(query,tf.transpose(keys,perm=[0,2,1]))/tf.math.sqrt(tf.cast(keys.shape[-1],dtype=tf.float32)),axis=-1)#(bnh,td,te)
        context = tf.matmul(att_m,values)#(bnh,td,d/h)
        context = tf.transpose(tf.reshape(context,(-1,self.num_nodes,dec_out.shape[1],context.shape[-1])),perm=[0,2,1,3])#[bh,td,n,d/h]
        context = tf.concat(tf.split(context,self.num_heads,axis=0),axis=-1)#(b,td,n,d)
        context = self.dense1(context)#(b,td,n,d)
        return context,att_m#(b,td,n,d)&(bnh,td,te)

class Encoder(tf.keras.layers.Layer):
    def __init__(self,num_units,input_dim,adj_mx,diffusion_steps,num_nodes,embed_units,
                 output_dim,if_spatial=True,if_adp=True,training=True,**kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.embedding_layer = tf.keras.layers.Dense(num_units,activation='linear',kernel_initializer='glorot_uniform',use_bias=True)
        self.training = training
        self.dropout = tf.keras.layers.Dropout(0.05)
        self.aedrcll_enc = AE_DCRNNCell(num_units,num_units,adj_mx,diffusion_steps,num_nodes,embed_units,output_dim,if_spatial=if_spatial,
                                        if_adp=if_adp)
        self.encoder_rnn = tf.keras.layers.RNN(self.aedrcll_enc,return_sequences=True,return_state=True)
        self.num_nodes = num_nodes
        self.num_units = num_units
    def call(self,x):
        _,t,n_d = x.shape
        x = self.dropout(tf.nn.leaky_relu(self.embedding_layer(tf.reshape(x,[-1,t,self.num_nodes,self.input_dim]))),training=self.training)
        x = tf.reshape(x,[-1,t,self.num_nodes*self.num_units])
        outputs,state = self.encoder_rnn(x)
        adp_enc = self.aedrcll_enc.adp
        return outputs,state,adp_enc#返回[b,t,nd],[b,nd],[n,n]

class Decoder(tf.keras.layers.Layer):
    def __init__(self,num_units,input_dim,adj_mx,diffusion_steps,num_nodes,embed_unit,output_dim,num_heads,horizen,
                 if_spatial=True,if_adp=True,training=True,**kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.embedding_layer = tf.keras.layers.Dense(num_units,activation='linear',kernel_initializer='glorot_uniform',use_bias=True)
        self.dropout = tf.keras.layers.Dropout(0.05)
        self.training = training
        self.aedrcll_dec = AE_DCRNNCell(num_units,num_units,adj_mx,diffusion_steps,num_nodes,embed_unit,
                                        output_dim,if_spatial=if_spatial,if_adp=if_adp)#定义了一次参数仅使用一次,输入张量concat上一步的更新隐层
        self.inter_attention = Multi_head_inter_attention(num_nodes,num_units,num_heads,input_len=12,horizen=horizen)
        self.num_nodes = num_nodes
        self.num_unis = num_units
        self.output_dim = output_dim
        self.embedding_units = num_units
        self.update_gate = tf.keras.layers.Dense(num_units)
        self.dense_out = tf.keras.layers.Dense(num_units)
        self.dense_layer = tf.keras.layers.Dense(output_dim,activation='linear',use_bias=True,kernel_initializer='glorot_uniform')
    def call(self,dec_inp,enc_out,h_prev,step,h_init_t=None):#传入单步[b,nd0],enc_out[B,T,nd1],h_prev最后一步的隐层状态[b,nd1],h_prev要传入一个list,换成Bahdanau 注意力的形式
        #0.embedding非线性映射dec_inp
        dec_inp =self.dropout(tf.nn.leaky_relu(self.embedding_layer(tf.reshape(dec_inp,[-1,self.num_nodes,self.input_dim]))),training=self.training)#b,1,n,d
        dec_inp = tf.expand_dims(dec_inp,axis=1)
        #1.运行循环单元,考虑了顺序关系
        dec_inp = tf.reshape(dec_inp,[-1,self.num_nodes*self.num_unis])
        if h_init_t==None:
            dec_inp = dec_inp
        else:#是否需要设置门控
            dec_inp = dec_inp+h_init_t
        outputs,h_prev_ = self.aedrcll_dec(dec_inp,h_prev)#-->[B,nd]&list[B,nd]
        #2.计算注意力机制
        context,att_m = self.inter_attention(enc_out,tf.expand_dims(h_prev_[0],axis=1),step)#(b,td,n,d)&#(bnh,td,te)
        #3.拼接&维度对齐
        h_t = tf.expand_dims(tf.reshape(h_prev_[0],[-1,self.num_nodes,self.num_unis]),axis=1)#[B,td,n,d]
        h_t = tf.concat([context,h_t],axis=-1)#[B,td,n,2d]
        h_init_t = tf.nn.tanh(self.dense_out(h_t))#[B,td,n,d]
        #4.reshape
        h_init_t = tf.reshape(tf.squeeze(h_init_t,axis=1),[-1,self.num_nodes*self.num_unis])#[b,nd]
        outputs=self.dense_layer(tf.reshape(h_init_t,[-1,self.num_nodes,self.num_unis]))#[b,n,d_out]
        outputs = tf.reshape(outputs,[-1,self.num_nodes*self.output_dim])#[b,nd_out]
        return outputs,h_prev_,h_init_t,att_m#[b,ndout],list[b,nd],[b,nd],[n,n],(bnh,1,te)

    
class AEDRCNN_Model(tf.keras.Model):#用于训练
    def __init__(self,num_units,input_dim,adj_mx,diffusion_steps,num_nodes,embed_units,output_dim,
                 num_heads,horizen=12,if_spatial=True,if_adp=True,training=True,**kwargs):
        '''
        编解码模型参数对称的Seq2Seq
        :param num_units: 循环单元隐层数
        :param adj_mx: 邻接矩阵，实数
        :param diffusion_steps: 扩散步骤k-hops
        :param num_nodes: 节点数目
        :param adp_units: 自适应节点维度
        :param temp_units: 空间异质性临时维度
        :param output_dim: 输出维度
        :param kwargs:
        '''
        super().__init__(**kwargs)
        self.num_nodes = num_nodes
        self.num_units = num_units
        self.input_dim = input_dim
        self.if_training = training
        self.horizen = horizen
        self.encoder = Encoder(num_units,input_dim,adj_mx,diffusion_steps,num_nodes,embed_units,output_dim,
                               if_spatial=if_spatial,if_adp=if_adp,training=self.if_training)
        self.decoder = Decoder(num_units,input_dim,adj_mx,diffusion_steps,num_nodes,embed_units,output_dim,num_heads,horizen=horizen,
                               if_spatial=if_spatial,if_adp=if_adp,training=self.if_training)
    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        return tf.cast(k/(k+tf.exp(global_step/k)),tf.float32)

    def call(self,enc_inp,dec_inp,global_step):#多步骤输[b,te,nd],teaching forcing[b,td,nd]
        # _h_prev = None
        dec_inp = tf.cast(dec_inp,dtype=enc_inp.dtype)
        dec_inp = tf.concat([enc_inp[:,-1:,:],dec_inp[:,:-1,:]],axis=1)#用于teacher forcing->right shifted
        enc_outputs,h_prev,enc_adp = self.encoder(enc_inp)#->[b,te,nd],[b,nd],[n,n]
        self.decoder.aedrcll_dec.adp = enc_adp
        dec_inp = tf.transpose(dec_inp,[1,0,2])#->[td,b,ndin]
        _att_ms = tf.TensorArray(dtype=tf.float32,size=dec_inp.shape[0])#在call中循环必须用tensorarray存储临时张量
        _outputs = tf.TensorArray(dtype=tf.float32,size=dec_inp.shape[0])
        h_init_t = None
        threshold = self._compute_sampling_threshold(global_step,tf.cast(1000,tf.float32))
        outputs = tf.reshape(tf.reshape(dec_inp,[self.horizen,-1,self.num_nodes,self.input_dim])[0,:,:,0:1],[-1,self.num_nodes])
        if not isinstance(h_prev, list):  # 单独写decoder要警惕传入的隐层状态是列表，需要统一为列表模式
            h_prev = [h_prev]  # 这个地方可以考虑一下自注意力机制
        for _step in range(dec_inp.shape[0]):
            #scheduled sampling
            c = tf.random.uniform((), minval=0, maxval=1.)
            result = tf.cond(tf.less(c, threshold), lambda: dec_inp[_step], lambda: tf.reshape(tf.concat([tf.reshape(outputs,[-1,self.num_nodes,1]),
                                                                                                          tf.reshape(dec_inp[_step],[-1,self.num_nodes,self.input_dim])[...,1:]],axis=-1),[-1,self.num_nodes*self.input_dim]))
            outputs, h_prev, h_init_t, att_m = self.decoder(result, enc_outputs, h_prev,_step,h_init_t)#outputs->[b,nd_out]
            _att_ms = _att_ms.write(_step,att_m)
            _outputs = _outputs.write(_step,outputs)
            #outputs要与下一步的_step拼接
        att_mss = tf.transpose(tf.squeeze(_att_ms.stack(),axis=2),[1,0,2])#(bnh,td,te)
        outputs = tf.transpose(_outputs.stack(),[1,0,2])#[b,td,n]
        return outputs,enc_adp,att_mss#(b,td,n),(n,n),(n,n),(bhn,td,te)
    


# # # 测试
# def main():
#     adj_mx = np.ones((207, 207))
#     encoder_input = tf.random.uniform((8, 12, 207*2))
#     decoder_input= tf.random.uniform((8, 12, 207*2))
#     model = AEDRCNN_Model(num_units=64,input_dim=2, adj_mx=adj_mx, diffusion_steps=2, num_nodes=adj_mx.shape[0], 
#                           embed_units=10, output_dim=1,num_heads=4)
#     a, b,c,d= model(encoder_input, decoder_input)
#     print(a.shape,b.shape,c.shape,d.shape)
#     print(model.summary())
# if __name__ == '__main__':
#     main()
