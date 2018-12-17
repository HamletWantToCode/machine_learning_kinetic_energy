# projected gradient

import numpy as np 
import pickle
from statslib.main.kernel_ridge import KernelRidge
from statslib.tools.utils import rbfKernel, rbfKernel_gd, n_split
from GPML.main.utils import Preprocess
from statslib.main.cross_validation import Cross_validation
 
np.random.seed(9)

# data preprocessing
with open('/Users/hongbinren/Documents/program/MLEK/Burke/quantumX1D', 'rb') as f:
    data = pickle.load(f)
with open('/Users/hongbinren/Documents/program/MLEK/Burke/potentialX1D', 'rb') as f1:
    potential = pickle.load(f1)
nsamples = data.shape[0]
index = np.arange(0, nsamples, 1, dtype='int')
np.random.shuffle(index)
train_data, train_potential = data[index[:1001]], potential[index[:1001]]
test_data, test_potential = data[index[1001:]], potential[index[1001:]]

# N=1
train_single, train_potential_single = train_data[train_data[:, 0]==1], train_potential[train_data[:, 0]==1]
test_single, test_potential_single = test_data[test_data[:, 0]==1], test_potential[test_potential[:, 0]==1]

train_X, train_y, train_dy = train_single[:200, 2:], train_single[:200, 1], -train_potential_single[:200, 1:]
test_X, test_y, test_dy = test_single[:, 2:], test_single[:, 1], -test_potential_single[:, 1:]

pp = Preprocess()
pp.fit(train_X)
train_X_t, train_dy_t = pp.transform(train_X, train_dy)
test_X_t, test_dy_t = pp.transform(test_X, test_dy)

# cross-validation
Sigma = np.linspace(1, 100, 50)
Lambda = np.logspace(-14, -3, 50)
ss, ll = np.meshgrid(Sigma, Lambda)
Params = np.c_[ss.reshape((-1, 1)), ll.reshape((-1, 1))]
Error = []

for sigma, lambda_ in Params:
    gamma = 1.0/(2*sigma**2)
    kernel = rbfKernel(gamma)
    kernel_gd = rbfKernel_gd(gamma)
    model = KernelRidge(kernel, lambda_)
    kfold = n_split(5, 200, random_state=5)
    CV = Cross_validation(kfold, model, kernel_gd)
    error = CV.run(train_X_t, train_y[:, np.newaxis], train_dy_t)
    Error.append(error)
Error = np.array(Error).reshape(ss.shape)
x_min = Error.argmin(axis=0)
y_min = Error.argmin(axis=1)

with open('error_surface', 'wb') as f2:
    pickle.dump(Error, f2)

import matplotlib.pyplot as plt 
plt.contourf(ss, ll, Error, 40)
plt.plot(ss[x_min, y_min], ll[x_min, y_min], 'ko')
plt.semilogx()
plt.semilogy()
plt.show()

