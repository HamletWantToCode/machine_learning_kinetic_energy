# machine learning with GP

import pickle
import numpy as np
from statslib.main.gauss_process import Gauss_Process_Regressor
from statslib.tools.utils import rbfKernel, rbfKernel_gd, rbfKernel_hess, meanSquareError
from MLEK.main.utils import irfft

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
mean_y = np.mean(train_y)
mean_dy = np.mean(train_dy, axis=0)
train_dy -= mean_dy
train_y -= mean_y
train_y_ = np.r_[train_y, train_dy.reshape(-1)]
test_y -= mean_y

gamma = 8
kernel = rbfKernel(gamma)
kernel_gd = rbfKernel_gd(gamma)
kernel_hess =rbfKernel_hess(gamma)
model = Gauss_Process_Regressor(kernel, 1e-3, kernel_gd, kernel_hess)
model.fit(train_X, train_y_[:, np.newaxis])
predict_y, predict_yerr = model.predict(test_X)

err_y = meanSquareError(predict_y, test_y)
print(err_y)

K_star = np.c_[kernel_gd(test_X, train_X), kernel_hess(test_X, train_X)]
predict_dy = K_star @ model.coef_
predict_dy = predict_dy.reshape(test_dy.shape)

import matplotlib.pyplot as plt
X = np.linspace(0, 1, 100)
Vq_pred = predict_dy[5]
Vq_true = test_dy[5]
Vx_true = irfft(Vq_true, 100)
Vx_pred = irfft(Vq_pred, 100)
plt.plot(X, Vx_pred, 'b')
plt.plot(X, Vx_true, 'r')
plt.show()
