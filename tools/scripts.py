# -*- coding:utf-8 -*-
"""
Software:PyCharm
File: scripts.py
Institution: --- CSU&BUCEA ---, China
E-mail: obitowen@csu.edu.cn
Author：Yanjie Wen
Date：2023年04月21日
My zoom: https://github.com/YanJieWen
"""
import os.path
import os
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
#别惊讶，一些小工具而已
def calculate_random_walk_matrix(adj_mx):
    '''
    对邻接矩阵执行归一化操作
    :param adj_mx: array形式的邻接矩阵
    :return:归一化后的稀疏矩阵
    '''
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1.).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx
def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates
def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # 将 sin 应用于数组中的偶数索引（indices）；2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # 将 cos 应用于数组中的奇数索引；2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)

class SqureLoss(tf.keras.losses.Loss):
    def __init__(self,**kwargs):
        '''
        自定义一个均方误差损失
        :param kwargs:
        '''
        super().__init__(**kwargs)
    def call(self,y_hat,y):
        return tf.math.square(y_hat-y)/2.

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        '''
        动态学习率
        :param d_model: 隐层单元数
        :param warmup_steps: 模型预热步骤
        '''
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def masked_mse_np(preds, labels, null_val=np.nan):
    labels_ = labels.copy()
    labels_[np.where(labels_ < 1e-3)] = null_val
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels_)
        else:
            mask = np.not_equal(labels_, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        rmse = np.square(np.subtract(preds, labels_)).astype('float32')
        rmse = np.nan_to_num(rmse * mask)
        return np.mean(rmse)
def masked_mae_np(preds, labels, null_val=np.nan):
    labels_ = labels.copy()
    labels_[np.where(labels_ < 1e-3)] = null_val
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels_)
        else:
            mask = np.not_equal(labels_, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels_)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)
def masked_mape_np(preds, labels, null_val=np.nan):
    labels_ = labels.copy()
    labels_[np.where(labels_ < 1e-3)] = null_val
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels_)
        else:
            mask = np.not_equal(labels_, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(preds, labels_).astype('float32'), labels_))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)
def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))

def cal_metrics(y_hat,y,null_val=0):
    mae = masked_mae_np(y_hat,y,null_val)
    rmse = masked_rmse_np(y_hat,y,null_val)
    mape = masked_mape_np(y_hat,y,null_val)
    return mae,rmse,mape

def load_seq_data(data_dir,data_type):
    data_file = os.path.join(data_dir,'%s.npz'%data_type)
    return np.load(data_file)

def load_adj_mx(data_dir,adj_name):
    data_file = os.path.join(data_dir,adj_name)
    file = open(data_file,'rb')
    data = pickle.load(file)
    file.close()
    return data

def save_fig(fig_id,IMAGES_PATH,tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

class Accumulator:  #@save
    """在n个变量上累加,计数器"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
class Norm:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def fit_transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self,data):
        return data*self.std+self.mean