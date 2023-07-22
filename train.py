# -*- coding:utf-8 -*-
"""
Software:PyCharm
File: train.py
Institution: --- CSU&BUCEA ---, China
E-mail: obitowen@csu.edu.cn
Author：Yanjie Wen
Date：2023年04月23日
My zoom: https://github.com/YanJieWen
"""

import tensorflow as tf
import time
import argparse
import numpy as np

from tools.scripts import *
from tools.fs_logging import *
from model_supervisor import *


def main(args):
    data_s = time.time()
    logging = Logger(args.log_dir,'INFO','INFO')
    logging.info(f'===数据处理开始===')
    adj_mx = load_adj_mx(os.path.join('./data/',args.data_dir), args.adj_name)
    seq_data_train = load_seq_data(os.path.join('./data/',args.data_dir),args.data_type[0])
    seq_data_val = load_seq_data(os.path.join('./data/',args.data_dir), args.data_type[1])
    x_inp_train = seq_data_train['x']
    y_out_train = seq_data_train['y']
    x_inp_val = seq_data_val['x']
    y_out_val = seq_data_val['y']
    train_data = np.concatenate([x_inp_train, y_out_train], axis=1)
    val_data = np.concatenate([x_inp_val, y_out_val], axis=1)
    sca = Norm(np.mean(train_data[..., :1]),np.std(train_data[..., :1]))
    _, num_time, num_node, input_dim = train_data.shape
    train_data[..., :1] = sca.fit_transform(train_data[..., :1])
    val_data[..., :1] = sca.fit_transform(val_data[..., :1])

    train_data = np.reshape(train_data, (-1, num_time, num_node * input_dim))
    val_data = np.reshape(val_data, (-1, num_time, num_node * input_dim))

    train_dataset = tf.data.Dataset.from_tensor_slices((train_data[:, :args.input_len, :], train_data[:, -args.horizen:, :]))
    train_dataset = train_dataset.cache()  # for train dataset
    train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size=args.batch_size)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_data[:, :args.input_len, :], val_data[:, -args.horizen:, :]))
    val_dataset = val_dataset.batch(batch_size=args.batch_size)
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    logging.info(f'===数据处理一共花费{time.time()-data_s:.2f}s===')

    supervisor = Supervisor_Model(logging, sca, tol_step=args.tol_step, val_loss=np.inf, train_iter=train_dataset,
                                  test_iter=val_dataset,num_units=args.num_units,input_dim=input_dim,adj_mx=adj_mx, diffusion_steps=args.diffusion_steps, num_nodes=adj_mx.shape[0],
                                  embed_units=args.embed_units, output_dim=args.output_dim, ckpt_path=os.path.join('./data/',args.data_dir,args.ckpt_path),
                                  max_keep=args.max_keep, horizen=args.horizen, epoch_num=args.epoch_num, num_heads=args.num_heads,if_spatial=True, if_adp=True)
    supervisor.train()


if __name__ == '__main__':
    np.random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='PEMS_BAY/',help='交通节点数据文件夹')
    parser.add_argument('--data_type', type=list, default=['train','val'], help='数据集的类型')
    parser.add_argument('--adj_name', type=str, default='adj_mx.pkl', help='邻接图路径')
    parser.add_argument('--log_dir', type=str, default='train.log', help='训练阶段的日志文件')
    parser.add_argument('--input_len', type=int, default=12, help='输入步长')
    parser.add_argument('--horizen', type=int, default=12, help='输出步长')
    parser.add_argument('--tol_step', type=int, default=15, help='当超过这个值终止训练')
    parser.add_argument('--num_units', type=int, default=64, help='循环单元的隐层细胞数')
    parser.add_argument('--diffusion_steps', type=int, default=1, help='最大扩散步数')
    parser.add_argument('--embed_units', type=int, default=10, help='节点嵌入维度')
    parser.add_argument('--output_dim', type=int, default=1, help='输出维度')
    parser.add_argument('--ckpt_path', type=str, default='checkpoints/', help='权重保存的文件夹')
    parser.add_argument('--epoch_num', type=int, default=150, help='模型遍历所有数据集的次数')
    parser.add_argument('--batch_size', type=int, default=16, help='每一批数据的样本数')
    parser.add_argument('--max_keep', type=int, default=5, help='权重保存的最多次数')
    parser.add_argument('--num_heads', type=int, default=4, help='注意力头数')

    args = parser.parse_args()
    main(args)