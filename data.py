# 导入并处理数据
import numpy as np
import pandas as pd
import torch
import os
from sklearn import preprocessing

class Data():
    def __init__(self,feature_dir='train.csv',label_dir='train_label.csv'):
        self.feature_dir = feature_dir
        self.label_dir = label_dir

    def get_data(self):
        self.feature = pd.read_csv(self.feature_dir,dtype={'date':'str'})
        self.label = pd.read_csv(self.label_dir)
        self.feature.drop(columns=['ID','date'],inplace=True)   #删除ID和date列数据，date列数据可能有用，这里为了好处理，先删除该列数据
        # print (self.feature.dtypes)

    def get_standard_data(self):
        self.get_data()
        mean = self.feature.mean()
        self.feature.fillna(mean,inplace=True)
        feature = self.feature.values

        label = self.label['label'].values
        min_max_scaler = preprocessing.MinMaxScaler()
        feature = min_max_scaler.fit_transform(feature)

        if os.path.exists('feature.csv') == False:
            # fea.to_csv('feature.csv')
            np.savetxt('feature.csv',feature,fmt='%.2f',delimiter=',')
        if os.path.exists('label.csv') == False:
            # lab.to_csv('label.csv')
            np.savetxt('label.csv',label,fmt='%d',delimiter=',')
        return feature,label