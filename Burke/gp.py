import pickle
import numpy as np
# from GPML.main.optimize import MCSA_Optimize
from GPML.main.utils import Preprocess
from GPML.main.gauss_process import Gauss_Process_Regressor
from GPML.main.utils import rbfKernel, rbfKernel_gd, rbfKernel_hess

np.random.seed(8)

with open('/Users/hongbinren/Documents/program/MLEK/Burke/quantumX1D', 'rb') as f:
    data = pickle.load(f)
with open('/Users/hongbinren/Documents/program/MLEK/Burke/potentialX1D', 'rb') as f1:
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
test_X, test_y, test_dy = test_data_single[:, 2:], test_data_single[:, 1], -test_potential_single[:, 1:]

pp = Preprocess()
pp.fit(train_X, train_y, train_dy, 5)
train_X_t, train_y_t, train_dy_t = pp.transform(train_X, train_y, train_dy)
train_target = np.r_[train_y_t, train_dy_t.reshape(-1)]
test_X_t, test_y_t, test_dy_t = pp.transform(test_X, test_y, test_dy)
project_test_dy = test_dy_t*pp.var_dY
project_test_dy += pp.mean_dY
project_test_dy = np.sum(project_test_dy[:, :, np.newaxis]*pp.transfer_mat.T, axis=1)

# training performance on y and dy
gamma = 1./(2*48**2)
sigma = 1e-10
# optimizer = MCSA_Optimize(0.1, 2000, 1e-4, 10000, verbose=0)
model = Gauss_Process_Regressor(rbfKernel, None, gradient_on=True, kernel_gd=rbfKernel_gd, kernel_hess=rbfKernel_hess)
model.fit(train_X_t, train_target[:, np.newaxis], optimize_on=False, gamma=gamma, sigma=sigma)
y_pred, y_err = model.predict(test_X_t)
print(model.gamma_, model.sigma_)

predict_KE = y_pred*pp.var_Y + pp.mean_Y
err = (predict_KE - test_y)**2

gamma = model.gamma_
K_star = np.c_[rbfKernel_gd(gamma, test_X_t, train_X_t), rbfKernel_hess(gamma, test_X_t, train_X_t)]
predict_project_dy = (K_star @ model.coef_).reshape(test_dy_t.shape)
predict_project_dy *= pp.var_dY
predict_project_dy += pp.mean_dY
predict_dy = np.sum(predict_project_dy[:, :, np.newaxis]*pp.transfer_mat.T, axis=1)
err_dy = np.mean((predict_dy - project_test_dy)**2, axis=1)

import matplotlib.pyplot as plt
n_test = test_X.shape[0]
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
ax1.plot(np.arange(0, n_test, 1), err, 'b', label='KE')
ax1.plot(np.arange(0, n_test, 1), err_dy, 'g', label='project gd')
ax1.legend()

X = np.linspace(0, 1, 502)
ax2.plot(X, project_test_dy[20], 'r', label='project gd')
ax2.plot(X, predict_dy[20], 'b', label='predict gd')
ax2.legend()
plt.show()
