from data import Data
from sklearn import svm
import numpy as np

data = Data()
x,y = data.get_standard_data()
print (type(x))
print (type(y))
print (x)
print (x.shape[0])
print (y)
print (y.shape[0])

nan = np.isnan(x).any()
print (nan)
""" clf = svm.SVC(gamma='scale')
clf.fit(x,y)
predict_y = clf.predict(x)

print (predict_y) """
# print (len(np.argwhere(predict_y,y)))
