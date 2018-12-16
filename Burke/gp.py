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
train_X, train_y, train_dy = train_data_single[:200, 2:], train_data_single[:200, 1], -train_potential_single[:200, 1:]
n_train = train_X.shape[0]
Mean_X = np.mean(train_X, axis=0)
Cov = (train_X - Mean_X).T @ (train_X - Mean_X) / n_train
U, _, _ = np.linalg.svd(Cov)
train_X_t = train_X @ U[:, :5]
train_dy_t = train_dy @ U[:, :5]
scaler = np.amax(train_dy_t, axis=0)
np.maximum(1e-10, scaler, scaler)
train_dy_ts = train_dy_t / scaler
Mean_KE = np.mean(train_y)

test_X, test_y, test_dy = test_data_single[:, 2:], test_data_single[:, 1], -test_potential_single[:, 1:]
n_test = test_X.shape[0]
test_y -= mean_KE
test_X_t = test_X @ U[:, :5]
test_dy_t = test_dy @ U[:, :5]
test_dy_ts = test_dy_t / scaler
project_test_dy = np.sum(test_dy_ts[:, :, np.newaxis]*(U[:, :5]).T, axis=1)

# training performance on y and dy
# gamma = 1.27640384484
# sigma = 0.471508757686
params = np.array([0.3, 4.8])
sigma = 0.3
optimizer = MCSA_Optimize(0.1, 2000, 1e-4, 10000, verbose=1)
model = Gauss_Process_Regressor(rq_kernel, optimizer, gradient_on=True, kernel_gd=rq_kernel_gd, kernel_hess=rq_kernel_hess)
model.fit(train_X, train_y_[:, np.newaxis], optimize_on=False, params=params, sigma=sigma)
y_pred, y_err = model.predict(test_X)
print(model.gamma_, model.sigma_)

err = (y_pred - test_y)**2

gamma = model.gamma_
K_star = np.c_[rbfKernel_gd(gamma, test_X_t, train_X_t), rbfKernel_hess(gamma, test_X_t, train_X_t)]
predict_dy_t = (K_star @ model.coef_).reshape((n_test, 5))
predict_dy = np.sum(predict_dy_t[:, :, np.newaxis]*(U[:, :5]).T, axis=1)
err_dy = np.mean((predict_dy - project_test_dy)**2, axis=1)

# hyperparameter surface
# Gamma = np.linspace(0.01, 2.5, 40)
# Sigma = np.linspace(0.1, 1.0, 20)
# Alpha = np.linspace(0.1, 1.0, 10)
# L = np.linspace(1, 10, 10)
# aa, ll = np.meshgrid(Alpha, L)
# Parameters = np.column_stack((aa.reshape(-1), ll.reshape(-1)))
# energy_fval = []
# for a, l in Parameters:
#     val = model.log_likelihood(np.array([a, l, 0.3]), train_X, train_y_[:, np.newaxis], 1e-16)
#     energy_fval.append(val)
# energy_fval = np.array(energy_fval).reshape(aa.shape)

import matplotlib.pyplot as plt

fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
ax1.plot(np.arange(0, n_test, 1), err, 'b', label='KE')
ax1.plot(np.arange(0, n_test, 1), err_dy, 'g', label='project gd')
ax1.legend()

X = np.linspace(0, 1, 502)
ax2.plot(X, project_test_dy[32], 'r', label='project gd')
ax2.plot(X, predict_dy[32], 'b', label='predict gd')
ax2.legend()
plt.show()
