# 垃圾文件
import numpy as np
import torch 
import torch.utils.data as Data
from sklearn.preprocessing import Imputer
import pandas as pd

x = np.arange(1)
np.savetxt('a.csv',x,fmt='%d',delimiter=',')