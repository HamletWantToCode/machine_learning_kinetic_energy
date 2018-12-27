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
dens_X, Ek, Vx = np.fft.irfft(data[:, 1:], 100, axis=1)*100, data[:, 0].real, np.fft.irfft(Vq[:, 1:], 100, axis=1)*100
n = data.shape[0]
index = np.arange(0, n, 1, 'int')
np.random.shuffle(index)
train_X, train_y, train_dy = dens_X[index[:50]], Ek[index[:50]], -Vx[index[:50]]
test_X, test_y, test_dy = dens_X[index[50:]], Ek[index[50:]], -Vx[index[50:]]

gamma = 1e-3
lambda_ = 1e-11
flow = Workflow(5, gamma, lambda_, rbfKernel, rbfKernel_gd, KernelRidge)
flow.fit(train_X, train_y[:, np.newaxis])
pred_y, pred_dy = flow.predict(test_X)

err_y = np.mean((pred_y - test_y)**2)
err_dy = np.mean((pred_dy - test_dy)**2, axis=1)

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

ii = err_dy.argmax()
X = np.linspace(0, 1, 100)
ax3.plot(X, pred_dy[ii], 'b', label='predict')
ax3.plot(X, test_dy[ii], 'r', label='test')
ax3.set_xlabel('x')
ax3.set_ylabel('gradient')

plt.show()