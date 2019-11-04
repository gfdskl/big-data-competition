# 全连接神经网络算法
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Nn(nn.Module):
    def __init__(self,in_feature,out_feature):
        super(Nn,self).__init__()
        self.fc1 = nn.Linear(in_feature,30)
        self.fc2 = nn.Linear(30,out_feature)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x

