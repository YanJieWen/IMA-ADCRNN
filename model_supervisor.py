# -*- coding:utf-8 -*-
"""
Software:PyCharm
File: model_supervisor.py
Institution: --- CSU&BUCEA ---, China
E-mail: obitowen@csu.edu.cn
Author：Yanjie Wen
Date：2023年04月21日
My zoom: https://github.com/YanJieWen
"""
import numpy as np
import tensorflow as tf
from tools.scripts import *
from model import *
import time
from tools.fs_logging import *
from sklearn.preprocessing import StandardScaler

class Supervisor_Model():
    def __init__(self,logger,scarlar,tol_step,val_loss,train_iter,test_iter,num_units,input_dim,adj_mx,diffusion_steps,num_nodes,embed_units,output_dim,ckpt_path,
                 max_keep,horizen,epoch_num,num_heads,if_spatial=True,if_adp=True,if_training=True):
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.scarlar = scarlar
        self.tol_step = tol_step
        self.val_loss = val_loss
        self.model = AEDRCNN_Model(num_units=num_units, input_dim=input_dim,adj_mx=adj_mx, diffusion_steps=diffusion_steps, num_nodes=num_nodes, embed_units=embed_units, 
                                   horizen=horizen ,output_dim=output_dim,num_heads=num_heads,if_spatial=if_spatial,if_adp=if_adp,training=if_training)
        self.horizen  = horizen
        self.epoch_num = epoch_num
        self._logging = logger
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.num_units = num_units
        self.num_heads = num_heads
        self.diffusion_steps = diffusion_steps
        self.adj_mx = adj_mx
        self.embed_units = embed_units
        self.output_dim = output_dim
        self.if_spatial=if_spatial
        self.if_adp = if_adp
        # step2：定义损失指标
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.RootMeanSquaredError(name='Root error')
        # step3：学习率与优化器
        learning_rate = CustomSchedule(num_units)#计划学习率
        self.optimizer = tf.keras.optimizers.Adam(learning_rate,beta_1=0.9, beta_2=0.98,
                                             epsilon=1e-8,clipnorm=1.0)
        #step4:管理检查点
        ckpt = tf.train.Checkpoint(model=self.model,
                                   optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=max_keep)
        if self.ckpt_manager.latest_checkpoint:
            ckpt.restore(self.ckpt_manager.latest_checkpoint)
            self._logging.info('Latest checkpoint restored!!')
    #调用损失函数
        # self.loss = tf.keras.losses.MeanSquaredError()


    def loss_fn(self,labels,preds,null_val=0.):
        # labels_ = tf.where(labels<1.,null_val,labels)#反归一化的时候，这个玩意可能会不等于0，精度损失
        if np.isnan(null_val):
            mask = ~tf.math.is_nan(labels)
        else:
            mask = (labels!=null_val)
        mask = tf.cast(mask, tf.float32)
        mask /= tf.reduce_mean(mask)
        mask = tf.where(tf.math.is_nan(mask), tf.zeros_like(mask), mask)
        # loss = tf.square(tf.subtract(preds, labels))
        loss = tf.math.abs(preds-labels)
        loss = loss * mask
        loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
        return tf.reduce_mean(loss)
    #自定义的训练流程
    # @tf.function
    def _train(self,enc_inp,dec_inp,glob_step):
        
        with tf.GradientTape() as tape:
            y_hat,_,_ = self.model(enc_inp,dec_inp,glob_step)
            dec_inp  = tf.cast(dec_inp,dtype=y_hat.dtype)
            dec_inp = tf.reshape(dec_inp, (-1, y_hat.shape[1], self.num_nodes, self.input_dim))[..., 0]
            #inverse
            dec_inp = self.scarlar.inverse_transform(dec_inp)
            y_hat =  self.scarlar.inverse_transform(y_hat)
            #===========================================================================================================
            loss = self.loss_fn(dec_inp,y_hat)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # print(self.model.trainable_variables)#看一下有没有node_embedding这个张量
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        glob_step.assign_add(1.)
        self.train_loss(loss)
        self.train_accuracy(dec_inp, y_hat)
    @tf.function
    def evaluate(self,encoder_input,decoder_input):#单步推理,如果维度为1直接设置decoder_input=None
        '''
        如果输入信息不包含二手信息，即维度为1.那么可以令decoder_input=None.如果包含二手信息，那么在工程应用中，可以令decoder_input为全0矩阵
        和二手信息的拼接。
        :param encoder_input:
        :param decoder_input:
        :return:
        '''
        self.model.training = False
        enc_out,states_value,enc_adp= self.model.encoder(encoder_input)
        self.model.decoder.aedrcll_dec.adp = enc_adp
        target_value = encoder_input[:, -1, :]#(b,nd),单步预测在decoder中进行拼接
        stop_condition = False
        prediction = []
        _step = 0
        h_init_t = None
        num_dims = target_value.shape[-1]/self.num_nodes
        # target_value = tf.transpose(target_value, [1, 0, 2])
        if decoder_input is not None:#如果不添加辅助信息则为None
            decoder_input = tf.reshape(decoder_input,(-1,self.horizen,self.num_nodes,self.input_dim))#[b,t,n,2]
        while not stop_condition:
            # if _h_prev ==None:
            #     _h_prev = states_value
            if not isinstance(states_value, list):
                states_value= [states_value]
            # [b,nd],list[b,nd],[b,nd]
            outputs, states_value, h_init_t,_= self.model.decoder(target_value, enc_out, states_value,_step,h_init_t)#[b,nd_out]
            if num_dims>1:#加二手信息处理比较麻烦
                outputs_ = tf.expand_dims(outputs,axis=-1)#[b,n,1]
                target_value = tf.concat([outputs_,tf.cast(decoder_input[:,_step,:,1:],dtype=outputs_.dtype)],axis=-1)
                target_value = tf.reshape(target_value,(-1,self.num_nodes*self.input_dim))#[b,2n]
                _step += 1
            else:
                target_value = outputs
            prediction.append(tf.expand_dims(outputs,axis=1))
            if len(prediction) == self.horizen:
                stop_condition = True
        pred = tf.concat(prediction, axis=1)#[b,td,n]
        self.model.training = True
        return pred
    
    def train(self):
        self._logging.info('---Training begin---')
        num_bathes = len(self.train_iter)
        self._logging.debug('There are totel {} bathes/epoch'.format(num_bathes))
        _tol = 0
        glob_step = tf.Variable(0, trainable=False, dtype=tf.float32)
        for epoch in range(self.epoch_num):
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            epoch_time = []
            eval_metric = Accumulator(4)
            eval_metric.reset()
            for step,(enc_inp,dec_inp) in enumerate(self.train_iter):
                s_s_time = time.time()
                self._train(enc_inp,dec_inp,glob_step)
                s_nfe_train = time.time()-s_s_time
                epoch_time.append(s_nfe_train)
                if (step+1)%(num_bathes//5)==0 or step==num_bathes-1:#每次间隔num//5或者最后一个步长打印损失
                    print(f'Step{step}/Epoch{epoch}\tloss={self.train_loss.result():.3f},'f'\ttrain_rmse={self.train_accuracy.result():.3f}')
            #验证阶段
            _gt = []
            _pred = []
            for _,(x,y) in enumerate(self.test_iter):
                y_hat = self.evaluate(x,y).numpy()#最后可能一个数据点
                y_hat = self.scarlar.inverse_transform(y_hat)
                y = tf.reshape(y,(-1,y_hat.shape[1],self.num_nodes,self.input_dim))[...,0]
                y = self.scarlar.inverse_transform(y.numpy())
                _gt .append(y)
                _pred.append(y_hat)
            gt_ = np.concatenate(_gt,axis=0)
            pred_ = np.concatenate(_pred,axis=0)
            #     mae, rmse, mape = cal_metrics(y_hat,y)
            #     eval_metric.add(mae,rmse,mape,1)
            # _mae,_rmse,_mape = eval_metric[0]/eval_metric[-1],eval_metric[1]/eval_metric[-1],eval_metric[2]/eval_metric[-1]
            _mae,_rmse,_mape = cal_metrics(pred_,gt_)
            _avg_nfe = np.mean(epoch_time)
            print(f'Epoch={epoch}-Time(s/per batch)={_avg_nfe:.2f}'f'\tMAE={_mae:.3f},RMSE={_rmse:.3f},MAPE={_mape:.3f}')
            if _mae<self.val_loss:
                ckpt_path  = self.ckpt_manager.save()
                self._logging.info(f'Saving checkpoint for epoch {epoch} at {ckpt_path}'f'\tMAE={_mae:.2f}')
                self.val_loss = _mae
                _tol = 0
            else:
                _tol+=1
            if _tol>=self.tol_step:
                break
        self._logging.info(f'Congratulations, the training reached the number of early stops!')
#测试
# def main():
#     train_data = np.random.rand(150,24,10,2)
#     val_data = np.random.rand(100,24,10,2)
#     adj_mx = np.ones((10, 10))
#     ss = Norm(np.mean(train_data[..., :1]),np.std(train_data[..., :1]))
#     _, num_time, num_node, input_dim = train_data.shape
#     train_data[...,:1] = ss.fit_transform(train_data[..., :1])
#     val_data[...,:1] = ss.fit_transform(val_data[..., :1])
#     train_data = np.reshape(train_data, (-1, num_time, num_node * input_dim))
#     val_data = np.reshape(val_data, (-1, num_time, num_node * input_dim))

#     train_dataset = tf.data.Dataset.from_tensor_slices((train_data[:, :12, :], train_data[:, -12:, :]))
#     train_dataset = train_dataset.cache()  # for train dataset
#     train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size=32)
#     train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
#     val_dataset = tf.data.Dataset.from_tensor_slices((val_data[:, :12, :], val_data[:, -12:, :]))
#     val_dataset = val_dataset.batch(batch_size=32)
#     val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
#     logger = Logger('train.log','INFO','INFO')
#     supervisor = Supervisor_Model(logger,ss,tol_step=10,val_loss=np.inf,train_iter=train_dataset ,test_iter=val_dataset
#                                   ,num_units=64,input_dim=input_dim,adj_mx=adj_mx,diffusion_steps=2,num_nodes=adj_mx.shape[0],
#                                   embed_units=10,output_dim=1,ckpt_path='./checkpoints/',max_keep=5,
#                                   horizen=12,epoch_num=100,num_heads=4,if_spatial=True,if_adp=True)
#     supervisor.train()

# if __name__ == '__main__':
#     main()





