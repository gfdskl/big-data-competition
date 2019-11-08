import numpy as np
import pandas as pd

class SaveData():
    # 保存测试集预测结果
    def __init__(self,test_predict_proba,test_id,submission_dir=r'submission.csv'):
        self.submission_dir = submission_dir
        self.test_predict_proba = test_predict_proba
        self.test_id = test_id

    def save_data(self):
        test_id = self.test_id.reshape(-1,1).astype(np.object)         #防止在合并时被转化为float类型数据
        test_label = self.test_predict_proba[:,1]
        test_submission = np.c_[test_id,test_label]
        test_submission_pd = pd.DataFrame(test_submission)
        test_submission_pd.to_csv('submission.csv',index=False)
