# 从original_data中导入并预处理数据，并将处理后的结果返回到pre_data中
import numpy as np
import pandas as pd
import torch
import os
from sklearn import preprocessing

class ImportData():
    def __init__(self,train_dir=r'original_data/train.csv',train_label_dir=r'original_data/train_label.csv',test_dir=r'original_data/test.csv'):
        # 初始化文件路径
        self.train_dir = train_dir
        self.train_label_dir = train_label_dir
        self.test_dir = test_dir



    def import_data(self):
        # 导入原始数据
        self.train = pd.read_csv(self.train_dir)
        self.train_label = pd.read_csv(self.train_label_dir)
        self.test = pd.read_csv(self.test_dir)


        
    def preprocess_data(self):
        # 对数据进行预处理
        min_max_scaler = preprocessing.MinMaxScaler()
        self.import_data()
        test_id = self.test['ID'].values
        train_id = self.train['ID'].values
        self.change_feature_data()

        # 处理train数据
        mean = self.train.mean()
        self.train.fillna(mean,inplace=True)
        train = self.train.values
        train = min_max_scaler.fit_transform(train)

        # 处理train_label数据
        train_label = self.train_label['label'].values

        # 处理test数据
        mean = self.test.mean()
        self.test.fillna(mean,inplace=True)
        test = self.test.values
        test = min_max_scaler.fit_transform(test)

        # 保存数据
    # if os.path.exists(r'pre_data/train.csv') == False:
    #         np.savetxt(r'pre_data/train.csv',train,fmt='%.2f',delimiter=',')
    #     if os.path.exists(r'pre_data/train_label.csv') == False:
    #         np.savetxt(r'pre_data/train_label.csv',train_label,fmt='%d',delimiter=',')
    #     if os.path.exists(r'pre_data/test.csv') == False:
    #         np.savetxt(r'pre_data/test.csv',test,fmt='%.2f',delimiter=',')
    #     if os.path.exists(r'pre_data/train.csv') == False:
    #         np.savetxt(r'pre_data/train.csv',train,fmt='%.2f',delimiter=',')
    #     if os.path.exists(r'pre_data/train_id.csv') == False:
    #         np.savetxt(r'pre_data/train_id.csv',train_id,fmt='%d',delimiter=',')
    #     if os.path.exists(r'pre_data/test_id.csv') == False:
    #         np.savetxt(r'pre_data/test_id.csv',test_id,fmt='%d',delimiter=',')
        np.savetxt(r'pre_data/train.csv',train,fmt='%.2f',delimiter=',')
        np.savetxt(r'pre_data/train_label.csv',train_label,fmt='%d',delimiter=',')
        np.savetxt(r'pre_data/test.csv',test,fmt='%.2f',delimiter=',')
        np.savetxt(r'pre_data/train_id.csv',train_id,fmt='%d',delimiter=',')
        np.savetxt(r'pre_data/test_id.csv',test_id,fmt='%d',delimiter=',') 
        return train,train_label,test



    def change_feature_data(self):
        # 将train和test中的date特征转化为day和hour特征并加入原数据，删除train和test的ID和date特征(处理date特征))
        train_dt = pd.to_datetime(self.train.date,format='%Y/%m/%d %H:%M:%S')
        self.train['day'] = train_dt.dt.day
        self.train['hour'] = train_dt.dt.hour
        self.train.drop(columns=['ID','date'],inplace=True)

        test_dt = pd.to_datetime(self.test.date,format='%Y/%m/%d %H:%M:%S')
        self.test['day'] = test_dt.dt.day
        self.test['hour'] = test_dt.dt.hour
        self.test.drop(columns=['ID','date'],inplace=True)   
