# projected gradient

import numpy as np
import pickle
from statslib.main.kernel_ridge import KernelRidge
from statslib.tools.utils import rbfKernel, rbfKernel_gd
from statslib.main.workflow import Workflow

np.random.seed(9)

with open('/home/hongbin/Documents/project/MLEK/Burke/quantumX1D', 'rb') as f:
    data = pickle.load(f)
with open('/home/hongbin/Documents/project/MLEK/Burke/potentialX1D', 'rb') as f1:
    potential = pickle.load(f1)

ne = 1
data_ne1 = data[data[:, 0]==ne]
potential_ne1 = potential[potential[:, 0]==ne]

n_ne1 = data_ne1.shape[0]
index = np.arange(0, n_ne1, 1, 'int')
np.random.shuffle(index)

train_X, train_y, train_dy = data_ne1[index[:200], 2:], data_ne1[index[:200], 1], -potential_ne1[index[:200], 1:]
test_X, test_y, test_dy = data_ne1[index[200:], 2:], data_ne1[index[200:], 1], -potential_ne1[index[200:], 1:]

n_train = train_X.shape[0]
n_test = test_X.shape[0]
mean_X = np.mean(train_X, axis=0)
Cov = (train_X - mean_X).T @ (train_X - mean_X) / n_train
U, S, _ = np.linalg.svd(Cov)

dens_Xt = (train_X - mean_X) @ U[:, :20]

# correlation between n' and Ek
corr = []
for i in range(20):
    corr.append(np.corrcoef(dens_Xt[:, i], train_y)[1, 0])

# build workflow
gamma, lambda_ = 0.0016, 1e-11
Err_Y = []
Err_dY = []
RANK = 20
for i in range(1, RANK):
    model = Workflow(i, gamma, lambda_, rbfKernel, rbfKernel_gd, KernelRidge)
    model.fit(train_X, train_y[:, np.newaxis])
    pred_y, pred_dy_t = model.predict(test_X)
    test_dy_t = test_dy @ model.tr_mat_
    err_y = np.mean((pred_y - test_y)**2)
    Err_Y.append(err_y)
    err_dy = np.mean(np.mean((pred_dy_t - test_dy_t)**2, axis=1))
    Err_dY.append(err_dy)

rank = 13
tran_mat = U[:, rank]
train_Xt_1, test_Xt_1 = (train_X - mean_X) @ tran_mat, (test_X - mean_X) @ tran_mat
train_dyt_1, test_dyt_1 = train_dy @ tran_mat, test_dy @ tran_mat

kernel = rbfKernel(gamma)
kernel_gd = rbfKernel_gd(gamma)
model = KernelRidge(kernel, lambda_)
model.fit(train_Xt_1[:, np.newaxis], train_y[:, np.newaxis])
pred_dyt_1 = 501*kernel_gd(test_Xt_1[:, np.newaxis], train_Xt_1[:, np.newaxis]) @ model.coef_

project_pred_dy_1 = pred_dyt_1[:, np.newaxis] @ tran_mat[np.newaxis, :]
project_test_dy_1 = test_dyt_1[:, np.newaxis] @ tran_mat[np.newaxis, :]

workflow_2 = Workflow(rank, gamma, lambda_, rbfKernel, rbfKernel_gd, KernelRidge)
workflow_2.fit(train_X, train_y[:, np.newaxis])
_, pred_dyt_n = workflow_2.predict(test_X)

test_dyt_n = test_dy @ workflow_2.tr_mat_

project_pred_dy_n = np.sum(pred_dyt_n[:, :, np.newaxis]*workflow_2.tr_mat_.T, axis=1)
project_test_dy_n = np.sum(test_dyt_n[:, :, np.newaxis]*workflow_2.tr_mat_.T, axis=1)

import matplotlib.pyplot as plt
fig, ((ax1, ax2, ax3, ax7), (ax4, ax5, ax6, ax8)) = plt.subplots(2, 4, figsize=(20, 8))
# n_test = test_X.shape[0]
# test_index = np.arange(0, n_test, 1)
# ax1.plot(test_index, err_y, 'b', label='KE error')
# ax1.plot(test_index, err_dy, 'g', label='GD error')
# ax1.set_xlabel('test examples')
# ax1.set_ylabel('error')
# ax1.legend()

ax1.plot(np.arange(1, RANK, 1), Err_dY, 'bo-')
ax1.set_xlabel('dimension')
ax1.set_ylabel('Error')
ax1.semilogy()
ax1.set_title('Error in gradient')

ax2.plot(np.arange(1, RANK, 1), Err_Y, 'bo-')
ax2.set_xlabel('dimension')
ax2.set_ylabel('Error')
ax2.semilogy()
ax2.set_title('Error in Ek')

ax3.plot(test_Xt_1, test_dyt_1, 'ro', label='test')
ax3.plot(test_Xt_1, pred_dyt_1, 'bo', label='predict')
ax3.legend()

color = ['b', 'g', 'r', 'm', 'k']
for i in range(5):
    ax4.scatter(dens_Xt[:, i], train_y, c=color[i], label='dim-%d' %(i+1))
ax4.set_xlabel(r"$n'$")
ax4.set_ylabel('Ek')
ax4.legend()

ax5.plot(np.arange(1, 20, 1), corr[:-1], 'bo-')
ax5.set_xlabel('dimension')
ax5.set_ylabel('correlation')

X = np.linspace(0, 1, 502)
ax6.plot(X, project_test_dy_1[43], 'r', label='test')
ax6.plot(X, project_pred_dy_1[43], 'b', label='predict')
ax6.set_xlabel('x')

ax8.plot(X, project_test_dy_n[43], 'r', label='test')
ax8.plot(X, project_pred_dy_n[43], 'b', label='predict')

plt.show()
# # cross-validation
# Sigma = np.linspace(1, 100, 50)
# Lambda = np.logspace(-14, -3, 50)
# ss, ll = np.meshgrid(Sigma, Lambda)
# Params = np.c_[ss.reshape((-1, 1)), ll.reshape((-1, 1))]
# Error = []

# for sigma, lambda_ in Params:
#     gamma = 1.0/(2*sigma**2)
#     kernel = rbfKernel(gamma)
#     kernel_gd = rbfKernel_gd(gamma)
#     model = KernelRidge(kernel, lambda_)
#     kfold = n_split(5, 200, random_state=5)
#     CV = Cross_validation(kfold, model, kernel_gd)
#     error = CV.run(train_X_t, train_y[:, np.newaxis], train_dy_t)
#     Error.append(error)
# Error = np.array(Error).reshape(ss.shape)
# x_min = Error.argmin(axis=0)
# y_min = Error.argmin(axis=1)

# with open('error_surface', 'wb') as f2:
#     pickle.dump(Error, f2)

# import matplotlib.pyplot as plt
# plt.contourf(ss, ll, Error, 40)
# plt.plot(ss[x_min, y_min], ll[x_min, y_min], 'ko')
# plt.semilogx()
# plt.semilogy()
# plt.show()

# gamma = 0.0016973451852941056
# lambda_ = 6.551285568595496e-11
