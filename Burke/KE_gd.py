# compute gradient of KE functional

import pickle
import numpy as np
from statslib.tools.utils import rbfKernel, rbfKernel_gd
from statslib.main.kernel_ridge import KernelRidge

np.random.seed(8)

with open('/home/hongbin/Documents/project/MLEK/Burke/quantumX1D', 'rb') as f:
    data = pickle.load(f)
with open('/home/hongbin/Documents/project/MLEK/Burke/potentialX1D', 'rb') as f1:
    potential = pickle.load(f1)
n_samples = data.shape[0]
index = np.arange(0, n_samples, 1, 'int')
np.random.shuffle(index)
train_data, train_potential = data[index[:1001]], potential[index[:1001]]
test_data, test_potential = data[index[1001:]], potential[index[1001:]]

# N=1
train_single = train_data[train_data[:, 0]==1]
test_single, test_potential_single = test_data[test_data[:, 0]==1], test_potential[test_potential[:, 0]==1]

train_X, train_y = train_single[:200, 2:], train_single[:200, 1]
test_X, test_y, test_dy = test_single[:, 2:], test_single[:, 1], -test_potential_single[:, 1:]

n_train = train_X.shape[0]
n_test = test_X.shape[0]
mean_x = np.mean(train_X, axis=0)
Cov = (train_X - mean_x).T @ (train_X - mean_x) / n_train
U, _, _ = np.linalg.svd(Cov)
train_X_t = train_X @ U[:, :5]
test_X_t = test_X @ U[:, :5]

test_dy_t = test_dy @ U[:, :5]
project_test_dy = np.sum(test_dy_t[:, :, np.newaxis]*(U[:, :5]).T, axis=1)

# model construct
gamma = 0.01
Lambda = 1e-10
kernel = rbfKernel(gamma)
kernel_gd = rbfKernel_gd(gamma)
model = KernelRidge(kernel, Lambda)
model.fit(train_X_t, train_y[:, np.newaxis])

# predict
pred_y = model.predict(test_X_t)
K_gd = kernel_gd(test_X_t, train_X_t)
pred_dy_t = (K_gd @ model.coef_).reshape((n_test, 5))
pred_dy = np.sum(pred_dy_t[:, :, np.newaxis]*(U[:, :5]).T, axis=1)
# pred_dy = (K_gd @ model.coef_).reshape(test_dy.shape)
pred_dy *= 501

# performance
err_y = (pred_y-test_y)**2
err_dy = np.mean((pred_dy-project_test_dy)**2, axis=1)

# projector
# def project_gd(X, dY):
#     n_ = X.shape[0]
#     projected_dY = np.zeros_like(dY)
#     for i in range(n_):
#         x, dy = X[i], dY[i]
#         D = np.sqrt(np.sum((x[np.newaxis, :] - train_X)**2, axis=1))
#         index = np.argsort(D)
#         neighbor_index = index[:30]
#         neighbor_X = train_X[neighbor_index]
#         Diff = train_X - x[np.newaxis, :]
#         Cov = (Diff.T @ Diff)/30.0
#         _, S, Vt = np.linalg.svd(Cov)
#         coef = Vt[:5] @ dy
#         projected_dy = np.sum(coef[:, np.newaxis]*Vt[:5], axis=0)
#         projected_dY[i] = projected_dy
#     return projected_dY

# project_test_dy = project_gd(test_X, test_dy)
# project_pred_dy = project_gd(test_X, pred_dy)

# err_project_dy = np.mean((project_pred_dy-project_test_dy)**2, axis=1)

import matplotlib.pyplot as plt
n_test_sample = test_X.shape[0]
X = np.linspace(0, 1, 502)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
ax1.plot(np.arange(0, n_test_sample, 1), err_dy, 'g', label='original gd')
# ax1.plot(np.arange(0, n_test_sample, 1), err_project_dy, 'b', label='project gd')
ax1.plot(np.arange(0, n_test_sample, 1), err_y, 'y', label='KE')
ax1.legend()
ax2.plot(X, project_test_dy[34], 'r', label='project gd')
# ax2.plot(X, project_pred_dy[34], 'b', label='project pred')
ax2.plot(X, pred_dy[34], 'g', label='original pred')
ax2.legend()
plt.show()
