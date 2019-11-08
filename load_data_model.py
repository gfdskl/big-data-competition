from import_data import ImportData
import pandas as pd
import os
from xgboost import XGBClassifier
import pickle


def load_data(read=1):
    # 加载数据，如果之前已经生成预处理数据并且read=1，则直接读取数据，否则重新生成数据
    if os.path.exists(r'pre_data/train.csv') == False or read == 0:
        data = ImportData()
        train,train_label,test = data.preprocess_data()
    else:
        train = pd.read_csv(r'pre_data/train.csv').values
        train_label = pd.read_csv(r'pre_data/train_label.csv').values
        test = pd.read_csv(r'pre_data/test.csv').values
    return train,train_label,test

def load_model(train,train_label,read=1):
    # 加载模型，如果模型已经存在并且read=1，直接读取，否则使用XGBoost训练模型
    if os.path.exists(r'XGBoost.pickle') == False or read == 0:
        x = train
        y = train_label.reshape(-1,)            #将train_label转化成行向量
        # 训练模型
        model = XGBClassifier()
        model.fit(x,y)
        predict_y = model.predict(x)
        print (predict_y)
        with open(r'XGBoost.pickle','wb') as fw:
            pickle.dump(model,fw) 
    else:
        with open(r'XGBoost.pickle','rb') as fr:
            model = pickle.load(fr)
    return model