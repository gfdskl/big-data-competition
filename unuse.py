# 垃圾文件
import numpy as np
import torch 
import torch.utils.data as Data

x = torch.arange(12)
y = torch.arange(12)
# print (x)
# print (y)
dataset_ = Data.TensorDataset(x,y)
loader = Data.DataLoader(
    dataset=dataset_,
    batch_size=4,
    shuffle=True,
    # num_workers=2,
)

for epoch in range(3):
    for step,(a,b) in enumerate(loader):
        print ("step:"+str(step))
        print (x)
        print (y)




