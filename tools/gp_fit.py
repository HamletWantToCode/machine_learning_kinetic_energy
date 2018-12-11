# machine learning with GP

import pickle
import numpy as np
from statslib.main.gauss_process import Gauss_Process_Regressor
from statslib.tools.utils import rbfKernel, rbfKernel_gd, rbfKernel_hess, meanSquareError
from MLEK.main.utils import irfft

np.random.seed(3)
with open('/Users/hongbinren/Documents/program/MLEK/data_file/quantum', 'rb') as f:
    data = pickle.load(f)
with open('/Users/hongbinren/Documents/program/MLEK/data_file/potential', 'rb') as f1:
    potential = pickle.load(f1)
nsamples = data.shape[0]
index = np.arange(0, nsamples, 1, dtype='int')
np.random.shuffle(index)

# build train/test set
train_X, train_y, train_dy = data[index[:50], 1:], data[index[:50], 0], -potential[index[:50]]
test_X, test_y, test_dy = data[index[80:], 1:], data[index[80:], 0], -potential[index[80:]]
mean_y = np.mean(train_y)
# mean_dy = np.mean(train_dy, axis=0)
# train_dy -= mean_dy
train_y -= mean_y
train_y_ = np.r_[train_y, train_dy.reshape(-1)]
test_y -= mean_y

gamma = 1
kernel = rbfKernel(gamma)
kernel_gd = rbfKernel_gd(gamma)
kernel_hess =rbfKernel_hess(gamma)
model = Gauss_Process_Regressor(kernel, 1e-5, kernel_gd, kernel_hess)
model.fit(train_X, train_y_[:, np.newaxis])
predict_y, predict_yerr = model.predict(train_X)

## predict derivative and variance
# K = kernel(train_X)
# Kgd = kernel_gd(train_X)
# Khess = kernel_hess(train_X)
# predict_dy = (Kgd @ model.coef_).reshape(train_dy.shape)
# predict_dy_err = np.diag(Khess - Kgd @ np.linalg.pinv(K) @ Kgd.T).reshape(train_dy.shape)

# Kgd = kernel_gd(train_X)
# Khess = kernel_hess(train_X)
# K_star = np.c_[Kgd, Khess]
# predict_dy = (K_star @ model.coef_).reshape(train_dy.shape)

import matplotlib.pyplot as plt
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
plt.plot(train_y, predict_y, 'bo')
plt.plot(train_y, train_y, 'r')

# ax2.plot(train_dy[0], predict_dy[0], 'bo')
plt.show()

# err_y = meanSquareError(predict_y, test_y)
# print(err_y)

# K_star = np.c_[kernel_gd(test_X, train_X), kernel_hess(test_X, train_X)]
# predict_dy = K_star @ model.coef_
# predict_dy = predict_dy.reshape(test_dy.shape)

# err_dy = np.sum((predict_dy-test_dy)**2, axis=1)
# print(err_dy)

# import matplotlib.pyplot as plt
# X = np.linspace(0, 1, 100)
# dT_predict = irfft(predict_dy[55], 100)
# dT_true = irfft(test_dy[55], 100)
# plt.plot(X, dT_true, 'r')
# plt.plot(X, dT_predict, 'b')
# plt.show()

