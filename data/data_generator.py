# -*- coding:utf-8 -*-
"""
Software:PyCharm
File: data_generator.py
Institution: --- CSU&BUCEA ---, China
E-mail: obitowen@csu.edu.cn
Author：Yanjie Wen
Date：2023年04月23日
My zoom: https://github.com/YanJieWen
"""

import numpy as np
import pandas as pd
import os
import time
import pickle
import argparse

from tools.fs_logging import *

class Data_Factory():
    def __init__(self,logger,traffic_df_filename,distance_matrix_file,his_len,horizen,output_dir,output_adj_file,train_ratio,test_ratio,sigma,add_time_in_day=True,add_day_in_week=True):
        self._logging = logger
        self.tdf = traffic_df_filename
        self.dmf = distance_matrix_file
        self.his_len = his_len
        self.horizen = horizen
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.test_ratio  = test_ratio
        self.atid = add_time_in_day
        self.adiw = add_day_in_week
        self.oaf = output_adj_file
        self.sigma = sigma

        self.df = self.read_out_df(self.tdf)
        self._logging.info(f'序列包含{len(self.df)//288}天数据,'f'传感器网络一共包含{self.df.shape[-1]}个节点!')
        self.dis_df = self.read_out_dis_df(self.dmf)

    @staticmethod
    def read_out_df(data_path):
        try:
            if data_path.endswith('.h5'):
                df = pd.read_hdf(data_path)
                return df
        except:
            print('Error File not find!')
    @staticmethod
    def read_out_dis_df(data_path):
        return pd.read_csv(data_path,dtype={'from': 'str', 'to': 'str'})

    def gen_train_val_test(self):
        x_offsets = np.sort(np.concatenate((np.arange(-(self.his_len-1), 1, 1),)))
        y_offsets = np.sort(np.arange(1, self.horizen+1, 1))
        x,y = self.gen_graph_seq_io_data(x_offsets,y_offsets)
        self._logging.debug(f'x的形状为{x.shape},'f'y的形状为{y.shape}')
        num_samples = x.shape[0]
        num_test = round(num_samples * self.test_ratio)
        num_train = round(num_samples * self.train_ratio)
        num_val = num_samples - num_test - num_train
        x_train, y_train = x[:num_train], y[:num_train]
        x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
        x_test, y_test = x[-num_test:], y[-num_test:]
        for cat in ["train", "val", "test"]:
            _x, _y = locals()["x_" + cat], locals()["y_" + cat]
            print(cat, "x: ", _x.shape, "y:", _y.shape)
            np.savez_compressed(
                os.path.join(self.output_dir, "%s.npz" % cat),
                x=_x,
                y=_y,
                x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
                y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
            )

    def gen_graph_seq_io_data(self,x_offsets,y_offsets):
        num_samples, num_nodes = self.df.shape
        data = np.expand_dims(self.df.values, axis=-1)
        data_list = [data]
        if self.atid:
            time_ind = (self.df.index.values - self.df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")  # 统计一天中的时间
            time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
            data_list.append(time_in_day)
        if self.adiw:
            day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
            day_in_week[np.arange(num_samples), :, self.df.index.dayofweek] = 1  # 统计一周中的星期几
            data_list.append(day_in_week)
        data = np.concatenate(data_list, axis=-1,dtype=np.float32)
        x, y = [], []
        min_t = abs(min(x_offsets))
        max_t = abs(num_samples - abs(max(y_offsets)))
        for t in range(min_t, max_t):
            x_t = data[t + x_offsets, ...]# 输入
            y_t = data[t + y_offsets, ...]  # 输出
            x.append(x_t)
            y.append(y_t)
        x = np.stack(x, axis=0)
        y = np.stack(y, axis=0)
        return x,y

    def gen_adj_mx(self):
        sensor_idx = self.df.columns.values.astype(np.str)
        id_map = {str(value): id for id, value in enumerate(sensor_idx)}
        dist_mx = np.zeros((len(sensor_idx), len(sensor_idx)), dtype=np.float32)
        dist_mx[:] = np.inf
        for row in self.dis_df.values:
            if (row[0] not in sensor_idx) or (row[1] not in sensor_idx):
                continue
            dist_mx[id_map.get(row[0]), id_map.get(row[1])] = row[2]
        distances = dist_mx[~np.isinf(dist_mx)].flatten()
        std = distances.std()
        adj_mx = np.exp(-np.square(dist_mx / std))
        adj_mx[adj_mx < self.sigma] = 0
        self.adj_2_pickle(self.oaf,adj_mx)


    @staticmethod
    def adj_2_pickle(output_file,data):
        file = open(output_file,'wb')
        pickle.dump(data,file)
        file.close()


def main():
    parser = argparse.ArgumentParser()
    logger = Logger('data.log','INFO','INFO')
    parser.add_argument('--traffic_df_filename', dest='tdf',type=str, default='./METR_LR/metr-la.h5',
                        help='交通节点数据文件')
    parser.add_argument('--distance_matrix_file', dest='daf', type=str, default='./METR_LR/distances_la_2012.csv',
                        help='邻接矩阵距离信息')
    parser.add_argument('--his_len', dest='hh', type=int, default=12,
                        help='历史序列输入长度')
    parser.add_argument('--horizen', dest='horizen', type=int, default=12,
                        help='序列的输出长度')
    parser.add_argument('--output_dir', dest='output_dir', type=str, default='METR_LR/',
                        help='交通数据的存储文件夹')
    parser.add_argument('--output_adj_file', dest='output_adj_file', type=str, default='./METR_LR/adj_mx.pkl',
                        help='邻接矩阵存储文件夹')
    parser.add_argument('--sigma', dest='sigma', type=float, default=0.1,
                        help='保证邻接矩阵的稀疏性')
    parser.add_argument('--train_ratio', dest='tr', type=float, default=0.7,
                        help='训练数据比例')
    parser.add_argument('--test_ratio', dest='ter', type=float, default=0.2,
                        help='测试数据占比')

    args = parser.parse_args()
    data_generator = Data_Factory(logger,args.tdf,args.daf,
                                  args.hh,args.horizen,args.output_dir,args.output_adj_file,args.tr,args.ter,args.sigma,
                                  add_time_in_day=True,add_day_in_week=False)#当add_day_in_week设置为True张量维度变深，可能CPU内存溢出
    data_generator.gen_train_val_test()
    data_generator.gen_adj_mx()
    data_name = args.tdf.split('/')[1]
    logger.info(f'数据集{data_name}被成功保存!')

if __name__ == '__main__':
    main()


