# gradient error

import pickle
import numpy as np
from statslib.main.workflow import Workflow
from statslib.main.kernel_ridge import KernelRidge
from statslib.tools.utils import rbfKernel, rbfKernel_gd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.ticker import FixedFormatter

np.random.seed(3239)
with open('quantumX1D', 'rb') as f:
    data = pickle.load(f)
with open('potentialX1D', 'rb') as f1:
    potential = pickle.load(f1)
ne = 4
dens_X, Ek, dEk_X = data[data[:, 0]==ne, 2:], data[data[:, 0]==ne, 1], -potential[data[:, 0]==ne, 1:]
n = dens_X.shape[0]
index = np.arange(0, n, 1, 'int')
np.random.shuffle(index)
train_X, train_y, train_dy = dens_X[index[:300]], Ek[index[:300]], dEk_X[index[:300]]
test_X, test_y, test_dy = dens_X[index[300:]], Ek[index[300:]], dEk_X[index[300:]]

gamma, lambda_ = 0.0009102982, 1.09854114e-10

# Error_dy = []
# fig1 = plt.figure(1, (5, 5))
# ax = fig1.gca()

# for i in range(1, 20):
#     model = Workflow(i, gamma, lambda_, rbfKernel, rbfKernel_gd, KernelRidge)
#     model.fit(train_X, train_y[:, np.newaxis])
#     _, pred_dyt = model.predict(test_X)
#     test_dyt = test_dy @ model.tr_mat_
#     err_dy = np.mean(np.mean((test_dyt - pred_dyt)**2, axis=1))
#     Error_dy.append(err_dy)
# ax.plot(np.arange(1, 20, 1), Error_dy, 'bo-')
# ax.set_xlabel('dimension')
# ax.set_ylabel('error')
# ax.semilogy()

# fig2 = plt.figure(2, (8, 8))
# grids = ImageGrid(fig2, 111, nrows_ncols=(3, 3), axes_pad=0.1, aspect=False)
# X = np.linspace(0, 1, 502)
# n_cmp = [1, 2, 3, 7, 8, 9, 13, 14, 15]
# for i, n in enumerate(n_cmp):
#     model = Workflow(n, gamma, lambda_, rbfKernel, rbfKernel_gd, KernelRidge)
#     model.fit(train_X, train_y[:, np.newaxis])
#     _, pred_dyt = model.predict(test_X)
#     test_dyt = test_dy @ model.tr_mat_
#     project_pred_dy = np.sum(pred_dyt[:, :, np.newaxis]*model.tr_mat_.T, axis=1)
#     project_test_dy = np.sum(test_dyt[:, :, np.newaxis]*model.tr_mat_.T, axis=1)
#     mean_project_pred_dy = np.mean(project_pred_dy, axis=0)
#     mean_project_test_dy = np.mean(project_test_dy, axis=0)
#     l1, l2 = grids[i].plot(X, mean_project_pred_dy, 'b-', X, mean_project_test_dy, 'r--')
# fig2.legend((l1, l2), ('predict', 'test'), 'upper right')
# fig2.text(0.5, 0.04, 'x', ha='center')
# fig2.text(0.04, 0.5, r'$\frac{\delta T}{\delta n(x)}$', va='center', rotation='vertical')

model = Workflow(20, gamma, lambda_, rbfKernel, rbfKernel_gd, KernelRidge)
model.fit(train_X, train_y[:, np.newaxis])
pred_y, pred_dyt = model.predict(test_X)
test_dyt = test_dy @ model.tr_mat_
test_Xt = (test_X - model.mean_X) @ model.tr_mat_

fig3, axes = plt.subplots(3, 3, figsize=(8, 8))
n_cmp = [1, 2, 3, 6, 7, 8, 11, 12, 13]
for i, n in enumerate(n_cmp):
    l1, l2 = axes[i//3, i%3].plot(test_Xt[:, n], test_dyt[:, n], 'ro', test_Xt[:, n], pred_dyt[:, n], 'bo')
fig3.legend((l1, l2), ('predict', 'test'), 'upper right')
fig3.text(0.5, 0.04, r"$n'(x)$", ha='center')
fig3.text(0.04, 0.5, r"$\frac{\delta T}{\delta n'(x)}$", va='center', rotation='vertical')
plt.show()



