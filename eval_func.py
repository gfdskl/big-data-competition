from sklearn.metrics import f1_score,accuracy_score

class EvalFunc():
    # 评估预测结果
    def __init__(self,label,predict):
        self.label = label
        self.predict = predict

    def my_f1_score(self):
        # 使用f1_score评估
        score = f1_score(self.label,self.predict)
        print ("f1_socre:"+str(score))
        return 
        
    def my_accuracy_score(self):
        # 使用accuracy_score评估
        score = accuracy_score(self.label,self.predict)
        print ("accuracy_score"+str(score))
        return 
