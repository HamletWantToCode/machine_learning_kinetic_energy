import pickle
import numpy as np
from statslib.tools.utils import rbfKernel, laplaceKernel

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

from statslib.tools.multitask_KRR import Multi_task_KRR
from statslib.main.kernel_ridge import KernelRidge
from MLEK.main.utils import irfft
from statslib.tools.utils import laplaceKernel_gd, rbfKernel_gd
gamma = 5
# kernel = rbfKernel(gamma)
# kernel_gd = rbfKernel_gd(gamma)
kernel = laplaceKernel(gamma)
kernel_gd = laplaceKernel_gd(gamma)
model = Multi_task_KRR(kernel, kernel_gd, 1e-5)
model.fit(train_X, train_y[:, np.newaxis], train_dy)
predict_y = model.predict(test_X)

model1 = KernelRidge(kernel, 1e-5)
model1.fit(train_X, train_y[:, np.newaxis])
predict_y1 = model1.predict(test_X)

err = np.mean((predict_y-test_y)**2)
err1 = np.mean((predict_y1-test_y)**2)
print(err)
print(err1)

def gradient(dens_q, model):
    inner_dens_q = dens_q[np.newaxis, :]
    # KM = kernel(inner_dens_q, train_X)
    alpha = model.coef_
    KM_gd = kernel_gd(inner_dens_q, train_X)
    # D_mat = inner_dens_q - train_X
    # D_div = np.sqrt((D_mat*D_mat.conj()).real)
    # np.maximum(1e-15, D_div, out=D_div)
    # gd_q = -gamma*np.sum(alpha[:, np.newaxis]*(D_mat/D_div)*KM.T, axis=0, dtype=np.complex64)
    # gd_q = -2*gamma*np.sum(alpha[:, np.newaxis] * (inner_dens_q - train_X) * KM.T, axis=0, dtype=np.complex64)
    gd_q = KM_gd @ alpha
    return np.squeeze(gd_q)

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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.plot(test_y, predict_y, 'bo')
ax1.plot(test_y, predict_y1, 'go', alpha=0.8)
ax1.plot(test_y, test_y, 'r')

predict_gd_q = gradient(test_X[31], model)
predict_gd_q1 = gradient(test_X[31], model1)
predict_gd_x = irfft(predict_gd_q, 100)
predict_gd_x1 = irfft(predict_gd_q1, 100)
X = np.linspace(0, 1, 100)
true_gd_q = test_dy[31]
true_gd_x = irfft(true_gd_q, 100)
ax2.plot(X, -true_gd_x, 'r')
ax2.plot(X, predict_gd_x, 'b')
ax2.plot(X, predict_gd_x1, 'g', alpha=0.8)
plt.show()
