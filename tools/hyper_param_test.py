import pickle
import numpy as np 
from statslib.main.kernel_ridge import KernelRidge
from statslib.tools.utils import rbfKernel, rbfKernel_gd
from statslib.main.workflow import Workflow
import matplotlib.pyplot as plt  

np.random.seed(53492)
with open('../data_file/quantum', 'rb') as f:
    data = pickle.load(f)
with open('../data_file/potential', 'rb') as f1:
    Vq = pickle.load(f1)
dens_X, Ek, Vx = np.fft.irfft(data[:, 1:], 500, axis=1)*500, data[:, 0].real, np.fft.irfft(Vq[:, 1:], 500, axis=1)*500
n = data.shape[0]
index = np.arange(0, n, 1, 'int')
np.random.shuffle(index)
train_X, train_y, train_dy = dens_X[index[:500]], Ek[index[:500]], -Vx[index[:500]]
test_X, test_y, test_dy = dens_X[index[500:]], Ek[index[500:]], -Vx[index[500:]]

gamma = 0.002682695795279722
lambda_ = 4.0949150623804276e-08

# gamma = 0.00019306977288832496
# lambda_ = 2.1209508879201927e-09
flow = Workflow(4, gamma, lambda_, rbfKernel, rbfKernel_gd, KernelRidge)
flow.fit(train_X, train_y[:, np.newaxis])
pred_y, pred_dy = flow.predict(test_X)

test_dyt = test_dy @ flow.tr_mat_
project_test_dy = test_dyt @ flow.tr_mat_.T

err_y = np.mean((pred_y - test_y)**2)
err_dy = np.mean((pred_dy - project_test_dy)**2, axis=1)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.plot(test_y, test_y, 'r', label='test')
ax1.plot(test_y, pred_y, 'bo', label='predict')
ax1.set_xlabel('test Ek')
ax1.set_ylabel('predict Ek')
ax1.set_title('predict error=%.3f' %(err_y))
ax1.legend()

n_test = test_X.shape[0]
ax2.stem(np.arange(0, n_test, 5), err_dy[::5], basefmt='C0-')
ax2.set_xlabel('index')
ax2.set_ylabel('error in gradient')
ax2.semilogy()

ii = err_dy.argmax()
X = np.linspace(0, 1, 500)
ax3.plot(X, pred_dy[ii], 'b', label='predict')
ax3.plot(X, project_test_dy[ii], 'r--', alpha=0.7, label='test')
ax3.set_xlabel('x')
ax3.set_ylabel('gradient')

plt.show()
