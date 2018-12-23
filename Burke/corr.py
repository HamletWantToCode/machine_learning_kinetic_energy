# various spatial distribution & prediction error

import pickle
import numpy as np  
from statslib.main.kernel_ridge import KernelRidge
from statslib.main.workflow import Workflow
from statslib.tools.utils import rbfKernel, rbfKernel_gd
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.ticker import NullFormatter, FixedLocator

with open('/Users/hongbinren/Documents/program/MLEK/Burke/quantumX1D', 'rb') as f:
    data = pickle.load(f)
with open('/Users/hongbinren/Documents/program/MLEK/Burke/potentialX1D', 'rb') as f1:
    potential = pickle.load(f1)
ne = 1
dens_X, Ek, dEk = data[data[:, 0]==ne, 2:], data[data[:, 0]==ne, 1], -potential[data[:, 0]==ne, 1:]
mean_X = np.mean(dens_X, axis=0)
n = dens_X.shape[0]

# spatial scale in each dimension
Cov = (dens_X - mean_X).T @ (dens_X - mean_X) / n
U, S, _ = np.linalg.svd(Cov)
dens_Xt = (dens_X - mean_X) @ U

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
color = ['b', 'g', 'r', 'c', 'm']
for i in range(5):
    ax.plot(dens_Xt[:, i], np.ones(n)*(i+1), color=color[i], marker='o', label='dim %d' %(i+1))
ax.set_xlabel('scale')
ax.yaxis.set_major_formatter(NullFormatter())
ax.legend()

fig1, ax1 = plt.subplots(1, 1, figsize=(5, 5))
Corr = []
for i in range(20):
    corr = np.corrcoef(dens_Xt[:, i], Ek)[1, 0]
    Corr.append(corr)
ax1.plot(np.arange(0, 20, 1), Corr, 'bo-')
ax1.set_xlabel('dimension')
ax1.xaxis.set_major_locator(FixedLocator(np.arange(0, 22, 2)))
ax1.set_ylabel('corrlation coefficient')

# machine learning
# np.random.seed(83223)
# index = np.arange(0, n, 1, 'int')
# np.random.shuffle(index)
# train_X, train_y, train_dy = dens_X[index[:250]], Ek[index[:250]], dEk[index[:250]]
# test_X, test_y, test_dy = dens_X[index[250:]], Ek[index[250:]], dEk[index[250:]]

# fig2 = plt.figure(2, (6, 6))
# grid1 = ImageGrid(fig2, rect=[0.1, 0.1, 0.8, 0.5], nrows_ncols=(1, 1), aspect=False)
# grid2 = ImageGrid(fig2, rect=[0.1, 0.7, 0.8, 0.2], nrows_ncols=(1, 4), axes_pad=0.1)

# gamma, lambda_ = 0.0009102982, 1.09854114e-10
# err = []
# for i in range(1, 5):
#     model = Workflow(i, gamma, lambda_, rbfKernel, rbfKernel_gd, KernelRidge)
#     model.fit(train_X, train_y[:, np.newaxis])
#     pred_y, pred_dyt = model.predict(test_X)
#     grid2[i-1].plot(test_y, pred_y, 'bo')
#     grid2[i-1].plot(test_y, test_y, 'r')
#     grid2[i-1].set_xlabel('test Ek')
#     grid2[i-1].set_ylabel('predict Ek')
#     err.append(np.mean((pred_y - test_y)**2))
#     # upp.append(np.amax((pred_y - test_y)**2))
#     # low.append(np.amin((pred_y - test_y)**2))
# err = np.array(err)
# grid1[0].plot(np.arange(1, 5, 1), err, 'bo-')
# grid1[0].set_xlabel('dimension')
# grid1[0].set_ylabel('error')
# grid1[0].xaxis.set_major_locator(FixedLocator(np.arange(1, 5, 1)))
plt.show()