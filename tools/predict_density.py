# ground state density

import numpy as np
import pickle
from statslib.main.kernel_ridge import KernelRidge
from statslib.tools.utils import rbfKernel, rbfKernel_gd
from MLEK.main.ground_state_density import Ground_state_density
import matplotlib.pyplot as plt

np.random.seed(83223)

with open('../data_file/quantum', 'rb') as f:
    data = pickle.load(f)
with open('../data_file/potential', 'rb') as f1:
    Vq = pickle.load(f1)
n = data.shape[0]
index = np.arange(0, n, 1, 'int')
np.random.shuffle(index)
dens_x, Ek, Vx = np.fft.irfft(data[:, 1:], 100, axis=1)*100, data[:, 0].real, np.fft.irfft(Vq[:, 1:], 100, axis=1)*100
train_X, train_y, train_dy = dens_x[index[:500]], Ek[index[:500]], -Vx[index[:500]]
test_X, test_y, test_dy = dens_x[index[500:]], Ek[index[500:]], -Vx[index[500:]]

gamma = 0.00019306977288832496
lambda_ = 2.1209508879201927e-09

n_train = train_X.shape[0]
mean_X = np.mean(train_X, axis=0)
Cov = (train_X - mean_X).T @ (train_X - mean_X) / n_train
U, _, _ = np.linalg.svd(Cov)
trans_mat = U[:, :2]
train_Xt, test_Xt = (train_X - mean_X) @ trans_mat, (test_X - mean_X) @ trans_mat
train_dyt, test_dyt = train_dy @ trans_mat, test_dy @ trans_mat

kernel = rbfKernel(gamma)
kernel_gd = rbfKernel_gd(gamma)
model = KernelRidge(kernel, lambda_)
model.fit(train_Xt, train_y[:, np.newaxis])
alpha = model.coef_

#######
Vx_example = -test_dy[324]
dens_example = test_X[324]
mu = 20
N = 1
eta = 1e-2
err = 1e-4

X = np.linspace(0, 1, 100)
dens_init = train_X[124]
plt.plot(X, dens_example, 'r', label='True')
plt.plot(X, dens_init, 'b', label='initial', alpha=0.6)

density = Ground_state_density(gamma, alpha, train_X, 2)
dens_optim = density.optimize(dens_init[np.newaxis, :], Vx_example, mu, N, eta, err, maxiters=10000)

plt.plot(X, dens_optim, 'g--', label='optimized')
plt.legend()
plt.xlabel('x')
plt.ylabel(r'$\rho(x)$')
plt.show()
