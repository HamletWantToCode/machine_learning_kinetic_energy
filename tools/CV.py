# predict value and gradient

import pickle
import numpy as np 
from statslib.main.workflow import Workflow
from statslib.main.kernel_ridge import KernelRidge
from statslib.tools.utils import rbfKernel, rbfKernel_gd, n_split
from statslib.main.cross_validation import Cross_validation

with open('/Users/hongbinren/Downloads/mnt/project/ML_periodic/quantum', 'rb') as f:
    data = pickle.load(f)
with open('/Users/hongbinren/Downloads/mnt/project/ML_periodic/potential', 'rb') as f1:
    potential = pickle.load(f1)
index = np.arange(0, data.shape[0], 1, 'int')
np.random.shuffle(index)
data_q, potential_q = data[:, 1:], potential
data_X, potential_X = np.fft.irfft(data_q, 100, axis=1), np.fft.irfft(potential_q, 100, axis=1)
train_X, train_y, train_dy = data_X[index[:1000]], data[index[:1000], 0].real, -potential_X[index[:1000]]
test_X, test_y, test_dy = data_X[index[2000:3000]], data[index[2000:3000], 0].real, -potential_X[index[2000:3000]]

Gamma, Lambda = np.logspace(-10, -5, 20), np.logspace(-10, -1, 20)
gg, ll = np.meshgrid(Gamma, Lambda)
Params = np.c_[gg.reshape((-1, 1)), ll.reshape((-1, 1))]
Error = []
for g, l in Params:
    workflow = Workflow(2, gamma, lambda_, rbfKernel, rbfKernel_gd, KernelRidge)
    kfold = n_split(10, 1000, random_state=5)
    CV = Cross_validation(kfold, workflow)
    error = CV.run(train_X, train_y[:, np.newaxis], train_dy)
    Error.append(error)
Error = np.array(Error).reshape(gg.shape)

import pickle
with open('data_file/error_surf', 'wb') as f1:
    pickle.dump(Error, f1)

x_min, y_min = Error.argmin()//20, Error.argmin()%20
print(gg[x_min, y_min])
print(ll[x_min, y_min])
print(Error[x_min, y_min])

import matplotlib.pyplot as plt 
plt.contourf(gg, ll, np.log(Error), 40)
plt.plot(gg[x_min, y_min], ll[x_min, y_min], 'ko')
plt.xlabel(r'$\gamma$')
plt.ylabel(r'$\lambda$')
plt.semilogx()
plt.semilogy()
plt.colorbar()
plt.savefig('data_file/error_surf.png')

    

