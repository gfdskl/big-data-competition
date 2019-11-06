# 垃圾文件
import numpy as np
import torch 
import torch.utils.data as Data
from sklearn.preprocessing import Imputer

x = [[1,2],[np.nan,3],[7,6]]
imp = Imputer(missing_values='NaN',strategy='mean',axis=0)
imp.fit(x)
x = imp.transform(x)
print (x)