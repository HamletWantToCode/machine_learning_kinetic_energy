import pickle
import numpy as np
from statslib.main.kernel_ridge import KernelRidge
from statslib.tools.utils import rbfKernel, rbfKernel_gd
from statslib.main.workflow import Workflow
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

np.random.seed(343822)
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

gamma = 0.002682695795279722
lambda_ = 4.0949150623804276e-08

X = np.linspace(0, 1, 100)

fig = plt.figure()
gds = ImageGrid(fig, 111, nrows_ncols=(3, 3), aspect=False, axes_pad=0.1)
dims = [1, 2, 3, 6, 7, 8, 11, 12, 13]
for i, d in enumerate(dims):
    flow = Workflow(d, gamma, lambda_, rbfKernel, rbfKernel_gd, KernelRidge)
    flow.fit(train_X, train_y[:, np.newaxis])
    pred_y, pred_dy = flow.predict(test_X)
    mean_pred_dy = np.mean(pred_dy, axis=0)
    project_test_dy = (test_dy @ flow.tr_mat_) @ flow.tr_mat_.T
    mean_test_dy = np.mean(project_test_dy, axis=0)
    l1, l2 = gds[i].plot(X, mean_pred_dy, 'b', X, mean_test_dy, 'r--')
fig.text(0.5, 0.04, 'x', ha='center')
fig.text(0.04, 0.5, r'$\frac{\delta T}{\delta n(x)}$', va='center', rotation='vertical')
fig.legend((l1, l2), ('predict', 'test'), 'upper right')
plt.show()

