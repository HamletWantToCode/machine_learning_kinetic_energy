import pickle
import numpy as np

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

from statslib.main.kernel_ridge import KernelRidge
from MLEK.main.utils import irfft
from statslib.tools.utils import rbfKernel, laplaceKernel

gamma = 0.01
kernel = rbfKernel(gamma)
model = KernelRidge(kernel, 1e-5)
model.fit(train_X, train_y[:, np.newaxis], cond=None)
predicty = model.predict(test_X)

err = np.mean((predicty-test_y)**2)
print(err)

# def gradient(dens_q, model):
#     inner_dens_q = dens_q[np.newaxis, :]
#     # KM = kernel(inner_dens_q, train_X)
#     alpha = model.coef_
#     KM_gd = kernel_gd(inner_dens_q, train_X)
#     # D_mat = inner_dens_q - train_X
#     # D_div = np.sqrt((D_mat*D_mat.conj()).real)
#     # np.maximum(1e-15, D_div, out=D_div)
#     # gd_q = -gamma*np.sum(alpha[:, np.newaxis]*(D_mat/D_div)*KM.T, axis=0, dtype=np.complex64)
#     # gd_q = -2*gamma*np.sum(alpha[:, np.newaxis] * (inner_dens_q - train_X) * KM.T, axis=0, dtype=np.complex64)
#     gd_q = KM_gd @ alpha
#     return np.squeeze(gd_q)

# from statslib.tools.multitask import Multi_task_Regressor
# from statslib.tools.multitask import Special_SGD
# optimizer = Special_SGD(1e-5, 1e-3, 1000, n_batch=200, verbose=1)
# gamma = 1
# kernel = laplaceKernel(gamma)
# model = Multi_task_Regressor(kernel, 1e-3, optimizer)
# model.fit(train_X, train_y[:, np.newaxis], train_dy, gamma)
# predict_y = model.predict(test_X)
# err = np.mean((predict_y - test_y)**2)
# print(err)

import matplotlib.pyplot as plt
plt.plot(test_y, predicty, 'bo')
plt.plot(test_y, test_y, 'ro', alpha=0.3)
plt.show()
