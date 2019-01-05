import pickle
import numpy as np 
from statslib.main.kernel_ridge import KernelRidge
from statslib.tools.utils import rbfKernel, rbfKernel_gd
from statslib.main.workflow import Workflow
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from matplotlib.ticker import *

np.random.seed(5439473)

with open('../data_file/quantum', 'rb') as f:
    data = pickle.load(f)
with open('../data_file/potential', 'rb') as f1:
    Vq = pickle.load(f1)
n = data.shape[0]
index = np.arange(0, n, 1, 'int')
np.random.shuffle(index)
dens_x, Ek, Vx = np.fft.irfft(data[:, 1:], 100, axis=1)*100, data[:, 0].real, np.fft.irfft(Vq[:, 1:], 100, axis=1)*100
train_X, train_y, train_dy = dens_x[index[:500]], Ek[index[:500]], -Vx[index[:500]]
test_X, test_y, test_dy = dens_x[index[500:]], Ek[index[500:]], -Vx[index[500:]]

gamma = 0.00019306977288832496
lambda_ = 2.1209508879201927e-09

# gamma = 0.002682695795279722
# lambda_ = 4.0949150623804276e-08

# error of KE with increasing dimension
fig1 = plt.figure(1)
gds = gridspec.GridSpec(5, 4)
ax1 = fig1.add_subplot(gds[0, 0])
ax2 = fig1.add_subplot(gds[0, 1])
ax3 = fig1.add_subplot(gds[0, 2])
ax4 = fig1.add_subplot(gds[0, 3])
ax5 = fig1.add_subplot(gds[1:, :4])
axes = [ax1, ax2, ax3, ax4]
Error = []
for i in range(1, 5):
    flow = Workflow(i, gamma, lambda_, rbfKernel, rbfKernel_gd, KernelRidge)
    flow.fit(train_X, train_y[:, np.newaxis])
    pred_y, pred_dy = flow.predict(test_X)
    axes[i-1].plot(test_y, pred_y, 'bo')
    axes[i-1].plot(test_y, test_y, 'r')
    axes[i-1].xaxis.set_major_formatter(NullFormatter())
    axes[i-1].yaxis.set_major_formatter(NullFormatter())
    err_y = np.mean((pred_y - test_y)**2)
    Error.append(err_y)
ax5.plot(np.arange(0, 4, 1), Error, 'bo-')
ax5.xaxis.set_major_locator(FixedLocator([0, 1, 2, 3]))
ax5.set_xlabel('dimension')
ax5.set_ylabel('error')

# error of gradient with increasing dimension
fig2 = plt.figure(2)
ax = fig2.gca()
Error_dy = []
for i in range(1, 20):
    flow = Workflow(i, gamma, lambda_, rbfKernel, rbfKernel_gd, KernelRidge)
    flow.fit(train_X, train_y[:, np.newaxis])
    pred_y, pred_dy = flow.predict(test_X)
    project_test_dy = (test_dy @ flow.tr_mat_) @ flow.tr_mat_.T
    err_dy = np.mean(np.mean((pred_dy - project_test_dy)**2, axis=1))
    Error_dy.append(err_dy)
ax.plot(np.arange(0, 19, 1), Error_dy, 'bo-')
ax.semilogy()
ax.set_xlabel('dimension')
ax.xaxis.set_major_locator(FixedLocator(np.arange(0, 19, 1)))
ax.set_ylabel('error')

plt.show()

