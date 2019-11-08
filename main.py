# 训练模型(XGBoost)
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import os
import pandas as pd
from xgboost import XGBClassifier

import load_data_model as ldm
from eval_func import EvalFunc as ef
from save_data import SaveData as sd

# 加载预处理数据和模型
train,train_label,test = ldm.load_data()
model = ldm.load_model(train,train_label)

# 测试训练集的准确度
train_predict = model.predict(train)
train_predict_proba = model.predict_proba(train)
ef1 = ef(train_label,train_predict)
ef1.my_f1_score()
ef1.my_accuracy_score()

# 预测test结果并且保存为csv文件
test_id = pd.read_csv(r'pre_data/test_id.csv').values
test_predict = model.predict(test)
test_predict_proba = model.predict_proba(test)
print ('test_predict_proba:')
print (test_predict_proba)
sd1 = sd(test_predict_proba,test_id)
sd1.save_data()


