import pickle
import numpy as np
from GPML.main.optimize import MCSA_Optimize
from GPML.main.gauss_process import Gauss_Process_Regressor
from GPML.main.utils import rq_kernel, rq_kernel_gd, rq_kernel_hess

np.random.seed(8)

with open('/home/hongbin/Documents/project/MLEK/Burke/quantumX1D', 'rb') as f:
    data = pickle.load(f)
with open('/home/hongbin/Documents/project/MLEK/Burke/potentialX1D', 'rb') as f1:
    potential = pickle.load(f1)
nsamples = data.shape[0]
index = np.arange(0, nsamples, 1, dtype='int')
np.random.shuffle(index)
train_data, train_potential = data[index[:1001]], potential[index[:1001]]
test_data, test_potential = data[index[1001:]], potential[index[1001:]]

# N=1
train_data_single, train_potential_single = train_data[train_data[:, 0]==1], train_potential[train_potential[:, 0]==1]
test_data_single, test_potential_single = test_data[test_data[:, 0]==1], test_potential[test_potential[:, 0]==1]
train_X, train_y, train_dy = train_data_single[:50, 2:], train_data_single[:50, 1], -train_potential_single[:50, 1:]
mean_KE = np.mean(train_y)
train_y -= mean_KE
scaler = np.amax(train_dy, axis=0, keepdims=True)
np.maximum(1e-10, scaler, out=scaler)
train_dy /= scaler
train_y_ = np.r_[train_y, train_dy.reshape(-1)]

test_X, test_y, test_dy = test_data_single[:, 2:], test_data_single[:, 1], -test_potential_single[:, 1:]
test_y -= mean_KE
test_dy /= scaler

# training performance on y and dy
# gamma = 1.27640384484
# sigma = 0.471508757686
params = np.array([0.3, 4.8])
sigma = 0.3
optimizer = MCSA_Optimize(0.1, 2000, 1e-4, 10000, verbose=1)
model = Gauss_Process_Regressor(rq_kernel, optimizer, gradient_on=True, kernel_gd=rq_kernel_gd, kernel_hess=rq_kernel_hess)
model.fit(train_X, train_y_[:, np.newaxis], optimize_on=False, params=params, sigma=sigma)
y_pred, y_err = model.predict(test_X)

acc = np.mean((y_pred - test_y)**2)
print(model.params_, model.sigma_)

params = model.params_
K_star = np.c_[rq_kernel_gd(params, test_X, train_X), rq_kernel_hess(params, test_X, train_X)]
predict_dy = K_star @ model.coef_
predict_dy = predict_dy.reshape(test_dy.shape)
average_performance_dy = np.mean((predict_dy - test_dy)**2, axis=1)

# hyperparameter surface
# Gamma = np.linspace(0.01, 2.5, 40)
# Sigma = np.linspace(0.1, 1.0, 20)
Alpha = np.linspace(0.1, 1.0, 10)
L = np.linspace(1, 10, 10)
aa, ll = np.meshgrid(Alpha, L)
Parameters = np.column_stack((aa.reshape(-1), ll.reshape(-1)))
energy_fval = []
for a, l in Parameters:
    val = model.log_likelihood(np.array([a, l, 0.3]), train_X, train_y_[:, np.newaxis], 1e-16)
    energy_fval.append(val)
energy_fval = np.array(energy_fval).reshape(aa.shape)

import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
# ax1.errorbar(test_y[::5], y_pred[::5], yerr=y_err[::5], fmt='o')
# ax1.plot(test_y, test_y, 'r')
# ax1.text(s='accuracy=%.3f' %(acc), x=test_y.min()+0.1*(test_y.max()-test_y.min()), y=test_y.min()+0.8*(test_y.max()-test_y.min()))

# # n_test = average_performance_dy.shape[0]
# # ax2.plot(np.arange(0, n_test, 1), average_performance_dy)
# X = np.linspace(0, 1, 22)
# # ax_in_ax2 = inset_axes(ax2, width='40%', height='40%', loc=1)
# # ax_in_ax2.plot(X, test_dy[20], 'r')
# # ax_in_ax2.plot(X, predict_dy[20], 'b--', alpha=0.5)
# ax2.plot(X, test_dy[20], 'r')
# ax2.plot(X, predict_dy[20], 'b--')

fig2 = plt.figure()
ax = fig2.gca()
surf = ax.contourf(aa, ll, energy_fval, 30)
ax.plot(params[0], params[1], 'ro')
ax.semilogx()
fig2.colorbar(surf, ax=ax)
plt.show()
