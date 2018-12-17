import pickle
import numpy as np

np.random.seed(3)

with open('/home/hongbin/Documents/project/MLEK/data_file/quantum', 'rb') as f:
    data = pickle.load(f)

with open('/home/hongbin/Documents/project/MLEK/data_file/potential', 'rb') as f1:
    potential = pickle.load(f1)

N = data.shape[0]
index = np.arange(0, N, 1, dtype='int')
np.random.shuffle(index)

train_X, train_y = data[index[:50], 1:], data[index[:50], 0].real
test_X, test_y, test_dy = data[index[100:], 1:], data[index[100:], 0].real, -potential[index[100:]]
mean = np.mean(train_y, axis=0)
train_y -= mean
test_y -= mean

from statslib.main.kernel_ridge import KernelRidge
from MLEK.main.utils import irfft
from statslib.tools.utils import rbfKernel, rbfKernel_gd

gamma = 1
kernel = rbfKernel(gamma)
kernel_gd = rbfKernel_gd(gamma)
model = KernelRidge(kernel, 1e-6)
model.fit(train_X, train_y[:, np.newaxis], cond=None)
predicty = model.predict(test_X)

err = np.mean((predicty-test_y)**2)
print(err)

def gradient(dens_q, model):
    N1, D = dens_q.shape
    N = train_X.shape[0]
    KM = kernel(dens_q, train_X)
    alpha = model.coef_
    KM_gd = (kernel_gd(dens_q, train_X)).reshape((N1*D, N))
#     # D_mat = inner_dens_q - train_X
#     # D_div = np.sqrt((D_mat*D_mat.conj()).real)
#     # np.maximum(1e-15, D_div, out=D_div)
#     # gd_q = -gamma*np.sum(alpha[:, np.newaxis]*(D_mat/D_div)*KM.T, axis=0, dtype=np.complex64)
    # gd_q = -2*gamma*np.sum(alpha[:, np.newaxis] * (inner_dens_q - train_X) * KM.T, axis=0, dtype=np.complex64)
    gd_q = KM_gd @ alpha
    return gd_q.reshape((N1, D))

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

dy_predict = gradient(test_X, model)
err_dy = np.sum((dy_predict - test_dy)**2, axis=1)
print(err_dy)

import matplotlib.pyplot as plt
X = np.linspace(0, 1, 100)
dT_predict = irfft(dy_predict[55], 100)
dT_true = irfft(test_dy[55], 100)
plt.plot(X, dT_true, 'r')
plt.plot(X, dT_predict, 'b')
plt.show()

