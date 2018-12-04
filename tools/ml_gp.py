# Gauss process regression 

import pickle
import numpy as np 
from statslib.tools.utils import rbfKernel, rbfKernel_gd, rbfKernel_hess

np.random.seed(23)

with open('/home/hongbin/Documents/project/MLEK/data_file/quantum', 'rb') as f:
    data = pickle.load(f)

with open('/home/hongbin/Documents/project/MLEK/data_file/potential', 'rb') as f1:
    potential = pickle.load(f1)

N = data.shape[0]
index = np.arange(0, N, 1, dtype='int')
np.random.shuffle(index)

train_X, train_y, train_dy = data[index[:40], 1:], data[index[:40], 0].real, potential[index[:40]]
test_X, test_y, test_dy = data[index[50:], 1:], data[index[50:], 0].real, potential[index[50:]]
mean = np.mean(train_y, axis=0)
train_y -= mean
test_y -= mean

gamma = 0.1
kernel = rbfKernel(gamma)
kernel_gd = rbfKernel_gd(gamma)
kernel_hess = rbfKernel_hess(gamma)
K = kernel(train_X)
K_gd = (kernel_gd(train_X)).reshape((40, -1))
K_hess = kernel_hess(train_X)
K_gauss_1 = np.hstack([K, K_gd])
K_gauss_2 = np.hstack([K_gd.T, K_hess])
K_gauss = np.vstack([K_gauss_1, K_gauss_2])
K_gauss += 0.001*np.eye(K_gauss.shape[0])

U, S, Vh = np.linalg.svd(K_gauss)
C = np.squeeze(U.T @ b) / S
alpha = np.sum(C[np.newaxis, :] * (Vh.T).conj(), axis=1)

K_predict = kernel(test_X, train_X)
K_gd_predict = kernel_gd(test_X, train_X)
K_gauss_predict = np.hstack([K_predict, K_gd_predict])
predict_y = K_gauss_predict @ alpha

import matplotlib.pyplot as plt 
plt.plot(test_y, predict_y, 'bo')
plt.plot(test_y, test_y, 'r')
plt.show()