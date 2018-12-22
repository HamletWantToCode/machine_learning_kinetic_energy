# projected gradient

import numpy as np 
import pickle
from sklearn.cluster import KMeans
from statslib.main.kernel_ridge import KernelRidge
from statslib.main.workflow import Workflow
from statslib.tools.utils import rbfKernel, rbfKernel_gd
# from statslib.main.cross_validation import Cross_validation
 
np.random.seed(9)

# data preprocessing
with open('/Users/hongbinren/Documents/program/MLEK/Burke/quantumX1D', 'rb') as f:
    data = pickle.load(f)
with open('/Users/hongbinren/Documents/program/MLEK/Burke/potentialX1D', 'rb') as f1:
    potential = pickle.load(f1)
nsamples = data.shape[0]
index = np.arange(0, nsamples, 1, dtype='int')
np.random.shuffle(index)
train_X, train_y, train_dy = data[index[:1001], 2:], data[index[:1001], 1], -potential[index[:1001], 1:]
test_X, test_y, test_dy = data[index[1001:], 2:], data[index[1001:], 1], -potential[index[1001:], 1:]

# clustering
cluster = KMeans(4, random_state=5)
train_labels = cluster.fit_predict(train_X)
test_labels = cluster.predict(test_X)
train_X_1, train_y_1, train_dy_1 = train_X[train_labels==0], train_y[train_labels==0], train_dy[train_labels==0]
test_X_1, test_y_1, test_dy_1 = test_X[test_labels==0], test_y[test_labels==0], test_dy[test_labels==0]

# build workflow
gamma = 0.0016973451852941056
lambda_ = 6.551285568595496e-11
pipe = Workflow(8, gamma, lambda_, rbfKernel, rbfKernel_gd, KernelRidge)
pipe.fit(train_X_1, train_y_1[:, np.newaxis])
pred_y_1, pred_dy_1 = pipe.predict(test_X_1)

project_test_dy_1 = np.sum((test_dy_1 @ pipe.tr_mat_)[:, :, np.newaxis]*pipe.tr_mat_.T, axis=1)
project_pred_dy_1 = np.sum(pred_dy_1[:, :, np.newaxis]*pipe.tr_mat_.T, axis=1)
err_y = (pred_y_1 - test_y_1)**2
err_dy = np.mean((project_pred_dy_1 - project_test_dy_1)**2, axis=1)

import matplotlib.pyplot as plt  
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(4, 8))
n_test = test_X_1.shape[0]
test_index = np.arange(0, n_test, 1)
ax1.plot(test_index, err_y, 'b', label='KE error')
ax1.plot(test_index, err_dy, 'g', label='GD error')
ax1.set_xlabel('test examples')
ax1.set_ylabel('error')
ax1.legend()

X = np.linspace(0, 1, 502)
ax2.plot(X, project_test_dy_1[45], 'r', label='True')
ax2.plot(X, project_pred_dy_1[45], 'b', label='Predict')
ax2.set_xlabel('x')
ax2.set_ylabel(r'$\frac{\delta T}{\delta n(x)}$')
ax2.legend()

ax3.plot(test_y_1, pred_y_1, 'bo')
ax3.plot(test_y_1, test_y_1, 'r')
plt.show()
# # cross-validation
# Sigma = np.linspace(1, 100, 50)
# Lambda = np.logspace(-14, -3, 50)
# ss, ll = np.meshgrid(Sigma, Lambda)
# Params = np.c_[ss.reshape((-1, 1)), ll.reshape((-1, 1))]
# Error = []

# for sigma, lambda_ in Params:
#     gamma = 1.0/(2*sigma**2)
#     kernel = rbfKernel(gamma)
#     kernel_gd = rbfKernel_gd(gamma)
#     model = KernelRidge(kernel, lambda_)
#     kfold = n_split(5, 200, random_state=5)
#     CV = Cross_validation(kfold, model, kernel_gd)
#     error = CV.run(train_X_t, train_y[:, np.newaxis], train_dy_t)
#     Error.append(error)
# Error = np.array(Error).reshape(ss.shape)
# x_min = Error.argmin(axis=0)
# y_min = Error.argmin(axis=1)

# with open('error_surface', 'wb') as f2:
#     pickle.dump(Error, f2)

# import matplotlib.pyplot as plt 
# plt.contourf(ss, ll, Error, 40)
# plt.plot(ss[x_min, y_min], ll[x_min, y_min], 'ko')
# plt.semilogx()
# plt.semilogy()
# plt.show()

# gamma = 0.0016973451852941056
# lambda_ = 6.551285568595496e-11