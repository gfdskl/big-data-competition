# 导入并处理数据
import numpy as np
import pandas as pd
import torch

class Data():
    def __init__(self,feature_dir='train.csv',label_dir='train_label.csv'):
        self.feature_dir = feature_dir
        self.label_dir = label_dir

    def get_data(self):
        self.feature = pd.read_csv(self.feature_dir)
        self.label = pd.read_csv(self.label_dir)

    def get_standard_data(self):
        self.get_data()
        # feature = self.feature.loc[:,list(self.feature.columns)[2:]]
        feature = self.feature.values[:,2:]
        # print (feature.head())
        # print (feature.shape[1])
        label = self.label['label'].values
        for i in range(feature.shape[1]):
            feature[:,i] = pd.cut(feature[:,i],100,labels=list(range(100)))
            # print (feature[i])
        return feature,label