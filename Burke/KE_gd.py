# compute gradient of KE functional

import pickle
import numpy as np
from statslib.tools.utils import rbfKernel, rbfKernel_gd
from statslib.main.kernel_ridge import KernelRidge

np.random.seed(8)

with open('/media/hongbin/Elements/project/Burke_paper/quantumX1D', 'rb') as f:
    data = pickle.load(f)
np.random.shuffle(data)
train_data = data[:1001]
test_data = data[1001:]

# N=1
# train_n = train_data[train_data[:, 0]==1]
# test_n = test_data[test_data[:, 0]==1]
train_X, train_y = train_data[:100, 2:], train_data[:400, 1]
mean_X = np.mean(train_X, axis=0, keepdims=True)
train_X -= mean_X
mean_KE = np.mean(train_y)
train_y -= mean_KE

gamma = 1.0/(2*47.1486636**2)
Lambda = 4.832930238571752e-11
kernel = rbfKernel(gamma)
kernel_gd = rbfKernel_gd(gamma)
model = KernelRidge(kernel, Lambda)
model.fit(train_X, train_y[:, np.newaxis])

def KE_gd(dens_X):
    N = len(dens_X)
    inner_dens_X = dens_X[np.newaxis, :]
    alpha = model.coef_
    KM_gd = kernel_gd(inner_dens_X, train_X)
    gd_X = KM_gd @ alpha
    return np.squeeze(gd_X)*(N-1)

# compute gradient
from main import compute, gaussPotential
a_low, a_high = 1, 10
b_low, b_high = 0.4, 0.6
c_low, c_high = 0.03, 0.1
A = np.random.uniform(a_low, a_high, 3)
B = np.random.uniform(b_low, b_high, 3)
C = np.random.uniform(c_low, c_high, 3)
N_electron = 3
N_grid = 500
new_data = compute(N_grid, N_electron, A, B, C)
new_dens_X = new_data[2:]
X = np.linspace(0, 1, N_grid+2, endpoint=True)
new_V = gaussPotential(A, B, C)
new_Vx = new_V(X)

import matplotlib.pyplot as plt
learn_dT = KE_gd(new_dens_X)
plt.plot(X, learn_dT, 'b', alpha=0.8)
plt.plot(X, -new_Vx, 'r')
plt.show()
