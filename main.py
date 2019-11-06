# 主函数，现在只把数据导入进去
from data import Data
from nn import Nn
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn

data = Data()
feature,label = data.get_standard_data()

feature = torch.from_numpy(feature)
label = torch.from_numpy(label)
print (feature)
feature_num = feature.size()[1]
print (feature_num)

net = Nn(feature_num,1)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

""" for epoch in range(10):
    runing_loss = 0.0 """
    



