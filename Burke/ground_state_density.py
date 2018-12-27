# ground state density

import numpy as np
import pickle
from statslib.main.kernel_ridge import KernelRidge
from statslib.tools.utils import rbfKernel, rbfKernel_gd
import matplotlib.pyplot as plt

np.random.seed(83223)

with open('quantumX1D', 'rb') as f:
    data = pickle.load(f)
with open('potentialX1D', 'rb') as f1:
    potential = pickle.load(f1)
ne = 4
dens_X, Ek, dEk = data[data[:, 0]==ne, 2:], data[data[:, 0]==ne, 1], -potential[data[:, 0]==ne, 1:]
index = np.arange(0, dens_X.shape[0], 1, 'int')
np.random.shuffle(index)

train_X, train_y, train_dy = dens_X[index[:250]], Ek[index[:250]], dEk[index[:250]]
test_X, test_y, test_dy = dens_X[index[250:]], Ek[index[250:]], dEk[index[250:]]
n_train = train_X.shape[0]
mean_X = np.mean(train_X, axis=0)
Cov = (train_X - mean_X).T @ (train_X - mean_X) / n_train
U, _, _ = np.linalg.svd(Cov)
trans_mat = U[:, :4]
train_Xt, test_Xt = (train_X - mean_X) @ trans_mat, (test_X - mean_X) @ trans_mat
train_dyt, test_dyt = train_dy @ trans_mat, test_dy @ trans_mat
project_test_dy = np.sum(test_dyt[:, :, np.newaxis]*trans_mat.T, axis=1)
# project_test_X = np.sum(test_Xt[:, :, np.newaxis]*trans_mat.T, axis=1)

gamma, lambda_ = 0.0009102982, 1.09854114e-10
kernel = rbfKernel(gamma)
kernel_gd = rbfKernel_gd(gamma)
model = KernelRidge(kernel, lambda_)
model.fit(train_Xt, train_y[:, np.newaxis])
alpha = model.coef_

#######
Vx_example = -potential[43, 1:]
dens_example = data[43, 2:]
mu = 1
N = 2
eta = 1e-2
err = 1e-4

# def energy(dens, mu, Vx):
#     dens_t = (dens - mean_X) @ trans_mat
#     Ek = model.predict(dens_t[np.newaxis, :])[0]
#     V = np.sum(Vx*dens)*(1.0/501)
#     lag_m = mu*(np.sum(dens)*(1.0/501) - 1)
#     return Ek + V - lag_m

# def energy_gd(coef, Vxt):
#     dens, mu = coef[:-1], coef[-1]
#     dens_t = (dens - mean_X) @ trans_mat
#     dEk = 501*(kernel_gd(dens_t[np.newaxis, :], train_Xt) @ alpha)
#     dV = Vxt
#     dlag_m = mu*np.ones(502) @ trans_mat
#     project_gd = (dEk + dV - dlag_m) @ trans_mat.T
#     gd_on_mu = (np.sum(dens)*(1./501) - ne)
#     return np.append(project_gd, gd_on_mu)

from MLEK.tools.ground_state_density import Ground_state_density

X = np.linspace(0, 1, 502)
dens_init = train_X[73]
# dens_init = np.ones(502, dtype=np.float64)
plt.plot(X, dens_example, 'r', label='True')
plt.plot(X, dens_init, 'b', label='initial', alpha=0.6)

density = Ground_state_density(gamma, alpha, train_X, 4)
dens_optim = density.optimize(dens_init[np.newaxis, :], Vx_example, mu, N, eta, err)

# E0 = energy(dens_init, mu, Vx_example)

# for i in range(1000):
#     gd = energy_gd(dens_init, mu, Vxt_example)
#     dens = dens_init - eta*gd
#     E1 = energy(dens, mu, Vx_example)
#     dens_init = dens
#     if abs(E1 - E0)<err:
#         print('convergence reached after %d of steps' %(i))
#         break
#     E0 = E1


# X = np.linspace(0, 1, 502)
# plt.plot(X, dens_example, 'r')
plt.plot(X, dens_optim, 'g--', label='optimized')
# plt.plot(X, project_test_X[216], 'm')
plt.legend()
plt.xlabel('x')
plt.ylabel(r'$\rho(x)$')
plt.show()
