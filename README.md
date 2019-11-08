# big-data-competition
## 数据预处理
1.对于train.csv和test.csv，从date特征中提取出__day__和__hour__特征，把ID特征提取到train_id.csv和test_id.csv中，并删除ID和date特征，对缺失值进行均值填充，最后使用MinMaxScaler进行特征缩放。

2.对于train_label.csv，删除ID特征。
##  文件和目录含义
* original_data：原始数据
    * test.csv：测试集特征
    * train_label.csv：训练集标签
    * train.csv：训练集特征
* pre_data：预处理后的数据
    * test_id.csv：训练集ID
    * test.csv：训练集特征(预处理)
    * train_id.csv：训练集ID
    * train_label.csv：训练集标签(预处理)
    * train.csv：训练集特征(预处理)
* 大数据算法赛教程.pdf：不用解释的吧。。。。
* eval_func.py：评估预测结果
* import_data.py：从原始数据中导入数据
* load_data_model.py：加载预处理数据和模型
* main.py：对test数据集预测标签
* save_data.py：保存test数据集预测结果
* submission.csv：test预测结果，最终要提交的文件
* unuse.py：垃圾文件，忽略即可
* XGBoost.pickle：XGBoost模型