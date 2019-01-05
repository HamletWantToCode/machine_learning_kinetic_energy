import pickle
import numpy as np 
from statslib.main.kernel_ridge import KernelRidge
from statslib.tools.utils import rbfKernel, rbfKernel_gd

with open('/home/hongbin/Documents/files/MLEK/quantum', 'rb') as f:
    data = pickle.load(f)
with open('/home/hongbin/Documents/files/MLEK/potential', 'rb') as f1:
    potential = pickle.load(f1)
dens_q, Ek, Vq = data[:, 1:], data[:, 0].real, potential[:, 1:]
Vq_extend = np.hstack([Vq, np.zeros((1000, 49))])
index = np.arange(0, dens_q.shape[0], 1, 'int')
np.random.shuffle(index)
train_dens, train_Ek, train_dEk = dens_q[index[:500]], Ek[index[:500]], -Vq_extend[index[:500]]
test_dens, test_Ek, test_dEk = dens_q[index[500:]], Ek[index[500:]], -Vq_extend[index[500:]]

n_cmp = 3
mean = np.mean(train_dens, axis=0)
Cov = ((train_dens - mean).conj()).T @ (train_dens - mean)
U, S, Vh = np.linalg.svd(Cov)
trans_mat = (Vh.conj()).T[:, :n_cmp]
train_X, test_X = (train_dens - mean) @ trans_mat, (test_dens - mean) @ trans_mat
train_y, test_y = train_Ek, test_Ek
train_dyt, test_dyt = train_dEk @ trans_mat, test_dEk @ trans_mat 

gamma = 0.001
lambda_ = 1e-10
kernel = rbfKernel(gamma)
model = KernelRidge(kernel, lambda_)
model.fit(train_X, train_y[:, np.newaxis])
pred_y = model.predict(test_X)

alpha = model.coef_
kernel_gd = rbfKernel_gd(gamma)
K_gd = kernel_gd(test_X, train_X)
pred_dyt = (K_gd @ alpha).reshape(test_X.shape)
project_pred_dy = pred_dyt @ (trans_mat.conj()).T 

project_test_dy = test_dyt @ (trans_mat.conj()).T 
err_y = (pred_y - test_y)**2
err_dy = np.mean((project_pred_dy - project_test_dy)**2, axis=1)

import matplotlib.pyplot as plt 
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.plot(test_y, test_y, 'r', label='true')
ax1.plot(test_y, pred_y, 'bo', label='predict')

X = np.linspace(0, 1, 100)
test_dyX = np.fft.irfft(project_test_dy, 100, axis=1)*100
pred_dyX = np.fft.irfft(project_pred_dy, 100, axis=1)*100
ax2.plot(X, test_dyX[54], 'r')
ax2.plot(X, pred_dyX[54], 'b--')

n_test = test_dens.shape[0]
ax3.plot(np.arange(0, n_test, 1), err_y, 'b')
ax3.plot(np.arange(0, n_test, 1), err_dy, 'g')
plt.show()

