# ground state density

import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from statslib.main.pipeline import MyPipe
from statslib.main.grid_search import MyGridSearchCV
from statslib.main.metric import make_scorer
# from statslib.main.kernel_ridge import KernelRidge
from statslib.main.gauss_process import GaussProcess
from statslib.main.pca import PrincipalComponentAnalysis
from statslib.main.utils import rbf_kernel, rbf_kernel_gradient, rbf_kernel_hessan

R = np.random.RandomState(328392)

with open('/Users/hongbinren/Documents/program/MLEK/Burke/quantumX1D', 'rb') as f:
    data = pickle.load(f)
with open('/Users/hongbinren/Documents/program/MLEK/Burke/potentialX1D', 'rb') as f1:
    potential = pickle.load(f1)
ne = 1
densx, Ek, dEk = data[data[:, 0]==ne, 2:], data[data[:, 0]==ne, 1], -potential[data[:, 0]==ne, 1:]
densx_train, densx_test, Ek_train, Ek_test, dEk_train, dEk_test = train_test_split(densx, Ek, dEk, test_size=0.4, random_state=R)

neg_mean_squared_error_scorer = make_scorer(mean_squared_error)
pipe = MyPipe([('reduce_dim', PrincipalComponentAnalysis()), ('regressor', GaussProcess(kernel=rbf_kernel, kernel_gd=rbf_kernel_gradient, gradient_on=True, kernel_hess=rbf_kernel_hessan))])
param_grid = [
              {
              'reduce_dim__n_components': [4],
              'regressor__sigma': [1.09854114e-4],
              'regressor__gamma': [0.0009102982]  
              }
             ]
grid_search = MyGridSearchCV(pipe, param_grid, cv=5, scoring=neg_mean_squared_error_scorer)
grid_search.fit(densx_train, Ek_train, dEk_train)
print(grid_search.cv_results_)

Ek_predict = grid_search.predict(densx_test)
project_dEk_predict = grid_search.predict_gradient(densx_test)

tr_mat = grid_search.best_estimator_.named_steps['reduce_dim'].tr_mat_
project_dEk_test = dEk_test @ tr_mat @ tr_mat.T

import matplotlib.pyplot as plt 
X = np.linspace(0, 1, 502)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(Ek_test, Ek_test, 'r')
ax1.plot(Ek_test, Ek_predict, 'bo')

ax2.plot(X, project_dEk_test[53], 'r')
ax2.plot(X, project_dEk_predict[53], 'b--')
plt.show()

# gamma, lambda_ = 0.0009102982, 1.09854114e-10
# pipe.fit(densx_train, Ek_train)
# Ek_predict = pipe.predict(densx_test)

# U = pipe.named_steps['reduce_dim'].tr_mat_
# project_dEk_test = dEk_test @ U @ U.T 
# project_dEk_predict = 501*pipe.predict_gradient(densx_test) @ U.T

#######
# Vx_example = -potential[43, 1:]
# dens_example = data[43, 2:]
# mu = 1
# N = 2
# eta = 1e-2
# err = 1e-4

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

# from MLEK.tools.ground_state_density import Ground_state_density

# X = np.linspace(0, 1, 502)
# dens_init = train_X[73]
# # dens_init = np.ones(502, dtype=np.float64)
# plt.plot(X, dens_example, 'r', label='True')
# plt.plot(X, dens_init, 'b', label='initial', alpha=0.6)

# density = Ground_state_density(gamma, alpha, train_X, 4)
# dens_optim = density.optimize(dens_init[np.newaxis, :], Vx_example, mu, N, eta, err)

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
# import matplotlib.pyplot as plt

# X = np.linspace(0, 1, 502)
# plt.plot(X, dens_example, 'r')
# plt.plot(X, dens_optim, 'g--', label='optimized')
# plt.plot(X, project_test_X[216], 'm')
# plt.legend()
# plt.xlabel('x')
# plt.ylabel(r'$\rho(x)$')
# plt.show()
