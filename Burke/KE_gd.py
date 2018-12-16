# projected gradient

import numpy as np 
import pickle
from statslib.main.kernel_ridge import KernelRidge
from statslib.tools.utils import rbfKernel, rbfKernel_gd
# from GPML.main.optimize import MCSA_Optimize

np.random.seed(9)

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
# train_data_single, train_potential_single = train_data[train_data[:, 0]==1], train_potential[train_potential[:, 0]==1]
# test_data_single, test_potential_single = test_data[test_data[:, 0]==1], test_potential[test_potential[:, 0]==1]
train_X, train_y = train_data[:400, 2:], train_data[:400, 1]
# mean_y = np.mean(train_y)
# train_y -= mean_y

test_X, test_y, test_dy = test_data[:, 2:], test_data[:, 1], -test_potential[:, 1:]
# test_y -= mean_y

# optimizer = MCSA_Optimize(0.1, 1000, 1e-4, 10000)
# model = Gauss_Process_Regressor(rbfKernel, optimizer)
# model.fit(train_X, train_y[:, np.newaxis], optimize_on=True)
# pred_y = model.predict(test_X)
gamma = 0.05
kernel = rbfKernel(gamma)
kernel_gd = rbfKernel_gd(gamma)
model = KernelRidge(kernel, 1e-9)
model.fit(train_X, train_y[:, np.newaxis])
pred_y = model.predict(test_X)

acc = np.mean((pred_y - test_y)**2)
print(acc)
K_gd_new_old = kernel_gd(test_X, train_X)
pred_dy = (K_gd_new_old @ model.coef_).reshape(test_dy.shape)
pred_dy *= 21

sample_X = test_X[13]
sample_dy = test_dy[13]
predict_dy = pred_dy[13]
D = np.sqrt(np.sum((sample_X[np.newaxis, :] - train_X)**2, axis=1))
index = np.argsort(D)
neighbor_index = index[:30]
neighbor_X = train_X[neighbor_index]
Diff = train_X - sample_X[np.newaxis, :]
Cov = (Diff.T @ Diff)/30.0

def project(X, y):
    _, S, Vt = np.linalg.svd(X)
    projected_coef = Vt[:5] @ y
    projected_y = np.sum(projected_coef[:, np.newaxis]*Vt[:5], axis=0)
    return projected_y

projected_sample_dy = project(Cov, sample_dy)
projected_pred_dy = project(Cov, predict_dy)

import matplotlib.pyplot as plt 
X = np.linspace(0, 1, 22)
plt.plot(X, projected_sample_dy, 'r--')
plt.plot(X, projected_pred_dy, 'b')
# plt.plot(X, predict_dy 'y', alpha=0.6)
plt.show()
