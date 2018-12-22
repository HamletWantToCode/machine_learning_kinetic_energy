# train - test error

import pickle
import numpy as np 
from statslib.main.kernel_ridge import KernelRidge
from statslib.tools.utils import rbfKernel
import matplotlib.pyplot as plt 

with open('/Users/hongbinren/Downloads/mnt/project/ML_periodic/quantum', 'rb') as f:
    data = pickle.load(f)
np.random.shuffle(data)
dens_q = data[:, 1:]
dens_q_2 = dens_q[data[:, 1].real > 1.8]
KE = data[:, 0].real
KE_2 = KE[data[:, 1].real > 1.8]
dens_X = np.fft.irfft(dens_q_2, 100, axis=1)*100

train_X, train_y = dens_X[:500], KE_2[:500]
test_X, test_y = dens_X[500:], KE_2[500:]

gamma = np.logspace(-10, 5, 50)
lambda_ = 0
Error_train = []
Error_test = []
for g in gamma:
    kernel = rbfKernel(g)
    model = KernelRidge(kernel, lambda_)
    model.fit(train_X, train_y[:, np.newaxis])
    pred_y_test = model.predict(test_X)
    pred_y_train = model.predict(train_X)
    Error_test.append(np.mean((test_y - pred_y_test)**2))
    Error_train.append(np.mean((train_y - pred_y_train)**2))

plt.plot(gamma, Error_train, 'b', label='train')
plt.plot(gamma, Error_test, 'g', label='test')
plt.semilogx()
plt.xlabel(r'$\gamma$')
plt.ylabel('error')
plt.show()