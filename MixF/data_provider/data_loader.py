import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_Arxiv(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                data_path='data/categories.xlsx',scale=True, freq='month'):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_excel(os.path.join(self.root_path,self.data_path))
        '''
        df_raw.columns: ['monthes', ...(categoriy_papers, category_authors)]
        '''
        cols = list(df_raw.columns)
        #删除掉时间这列
        cols.remove('monthes')
        # print(cols)
        # df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        #第一行的数据包含之前所有的数据信息(数据量较大，不适合作为预测数据信息)，所以这里选择删除掉
        df_raw=df_raw[1:]


        num_train = int(len(df_raw) * 0.8)
        num_test = int(len(df_raw) * 0.0)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        # print(df_raw)
        # print(cols)
        df_data = df_raw[cols]
        # print(df_data)
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        #得到train-vali-test的所以数据
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        # self.data_stamp = data['monthes']

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None, data_path='dataset/categories.xlsx',
                  scale=True, freq='monthes'):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.pred_len = size[1]
        # init
        assert flag in ['pred']

        self.scale = scale
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_excel(os.path.join(self.root_path,self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        #删除掉时间这行
        cols.remove('monthes')

        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)
        #取得非时间列的所有数据
        df_data = df_raw[cols]

        data = df_data.values
        
        self.time_data= df_raw['monthes']
        self.data_x = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        seq_x = self.data_x[s_begin:s_end]

        return seq_x

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
