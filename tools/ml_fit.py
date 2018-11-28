import pickle
import numpy as np 
from statslib.main.kernel_ridge import KernelRidge
from statslib.tools.utils import rbfKernel, laplaceKernel

with open('data_file/quantum', 'rb') as f:
    data = pickle.load(f)

with open('data_file/potential', 'rb') as f1:
    potential = pickle.load(f1)

N = data.shape[0]
index = np.arange(0, N, 1, dtype='int')
np.random.shuffle(index)

train_X, train_y = data[index[:200], 1:], data[index[:200], 0].real
test_X, test_y = data[index[300:], 1:], data[index[300:], 0].real
mean = np.mean(train_y, axis=0)
train_y -= mean
test_y -= mean

gamma = 1
kernel = laplaceKernel(gamma)
model = KernelRidge(kernel, 1e-3)
model.fit(train_X, train_y[:, np.newaxis])
predict_y = model.predict(test_X)

err = np.mean((predict_y-test_y)**2)
print(err)

alpha = model.coef_
Xi = model.X_fit_
def gradient(dens_q):
    D = dens_q[np.newaxis, :] - Xi
    manhattan = np.sqrt(D*D.conj())
    manhattan[:, 0] += 1e-8
    Kvector = np.squeeze(kernel(dens_q[np.newaxis, :], Xi))
    mat = alpha[:, np.newaxis] * D.conj() * Kvector[:, np.newaxis] / manhattan
    return -gamma*np.sum(mat, axis=0)
from MLEK.main.utils import irfft
test_i = 23

dTq = gradient(test_X[test_i])
dTx = irfft(dTq, 100)
X = np.linspace(0, 1, 100)

test_potential = potential[index[300:], 1:]
test_mu = test_potential[test_i, 0]
test_Vq = test_potential[test_i]
test_Vx = irfft(test_Vq, 100)

import matplotlib.pyplot as plt 
plt.plot(X, dTx, 'b')
plt.plot(X, test_mu - test_Vx, 'r')
# plt.plot(test_y, predict_y, 'bo')
# plt.plot(test_y, test_y, 'r')
plt.show()