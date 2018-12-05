# machine learning with GP 

import pickle
import numpy as np 
from statslib.main.gauss_process import Gauss_Process_Regressor
from statslib.tools.utils import rbfKernel, rbfKernel_gd, rbfKernel_hess, meanSquareError

np.random.seed(3)
with open('/home/hongbin/Documents/project/MLEK/data_file/quantum', 'rb') as f:
    data = pickle.load(f)
with open('/home/hongbin/Documents/project/MLEK/data_file/potential', 'rb') as f1:
    potential = pickle.load(f1)
nsamples = data.shape[0]
index = np.arange(0, nsamples, 1, dtype='int')
np.random.shuffle(index)

# build train/test set
train_X, train_y, train_dy = data[index[:50], 1:], data[index[:50], 0], potential[index[:50]]
test_X, test_y, test_dy = data[index[100:], 1:], data[index[100:], 0], potential[index[100:]]
mean = np.mean(train_y)
train_y -= mean
train_y_ = np.r_[train_y, train_dy.reshape(-1)]

gamma = 0.1
kernel = rbfKernel(gamma)
kernel_gd = rbfKernel_gd(gamma)
kernel_hess =rbfKernel_hess(gamma)
model = Gauss_Process_Regressor(kernel, 1e-3, kernel_gd, kernel_hess)
model.fit(train_X, train_y_[:, np.newaxis])
predict_y = model.predict(test_X)

err = meanSquareError(predict_y, test_y)
print(err)

import matplotlib.pyplot as plt 
plt.plot(test_y, predict_y, 'bo')
plt.plot(test_y, test_y, 'r')
plt.show()
