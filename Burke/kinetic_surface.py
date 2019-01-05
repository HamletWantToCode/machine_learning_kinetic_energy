# kinetic energy surface

import pickle
import numpy as np 
from statslib.main.kernel_ridge import KernelRidge
from statslib.tools.utils import rbfKernel
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import axes3d

np.random.seed(3239)
with open('quantumX1D', 'rb') as f:
    data = pickle.load(f)
ne = 2
dens_X, Ek = data[data[:, 0]==ne, 2:], data[data[:, 0]==ne, 1]
n = dens_X.shape[0]
index = np.arange(0, n, 1, 'int')
np.random.shuffle(index)
train_X, train_y = dens_X[index[:250]], Ek[index[:250]]
test_X, test_y = dens_X[index[250:]], Ek[index[250:]]
mean_X = np.mean(train_X, axis=0)
n_train = train_X.shape[0]
Cov = (train_X - mean_X).T @ (train_X - mean_X) / n_train
U, _, _ = np.linalg.svd(Cov)
train_Xt = (train_X - mean_X) @ U[:, :2]
test_Xt = (test_X - mean_X) @ U[:, :2]

gamma, lambda_ = 0.0009102982, 1.09854114e-10
kernel = rbfKernel(gamma)
model = KernelRidge(kernel, lambda_)
model.fit(train_Xt, train_y[:, np.newaxis])
pred_y = model.predict(test_Xt)
err_y = np.mean((pred_y - test_y)**2)
print(err_y)

dens_Xt = (dens_X - mean_X) @ U[:, :2]
x_max, y_max = np.amax(dens_Xt, axis=0)
x_min, y_min = np.amin(dens_Xt, axis=0)
X = np.linspace(x_min, x_max, 50)
Y = np.linspace(y_min, y_max, 50)
xx, yy = np.meshgrid(X, Y)
XY = np.c_[xx.reshape((-1, 1)), yy.reshape((-1, 1))]
zz = model.decisionFunction(XY).reshape(xx.shape)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(xx, yy, zz, alpha=0.5)
ax.scatter(dens_Xt[:, 0], dens_Xt[:, 1], Ek, c='r', label='density-KE')
ax.set_xlabel('principal #1')
ax.set_ylabel('principal #2')
ax.set_zlabel('Ek')
ax.legend()
plt.show()
