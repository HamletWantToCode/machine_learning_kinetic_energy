import pickle
import numpy as np
from sklearn.cluster import KMeans
from statslib.main.kernel_ridge import KernelRidge
from statslib.main.workflow import Workflow
from statslib.tools.utils import rbfKernel, rbfKernel_gd, n_split
from statslib.main.cross_validation import Cross_validation

# train - test split
np.random.seed(3)

with open('/home/hongbin/Documents/project/MLEK/data_file/quantum', 'rb') as f:
    data = pickle.load(f)
with open('/home/hongbin/Documents/project/MLEK/data_file/potential', 'rb') as f1:
    potential = pickle.load(f1)
index = np.arange(0, data.shape[0], 1)
np.random.shuffle(index)
train_Q, train_y, train_potential = data[index[:1001], 1:], data[index[:1001], 0].real, potential[index[:1001]]
test_Q, test_y, test_potential = data[index[1001:], 1:], data[index[1001:], 0].real, potential[index[1001:]]

train_X, train_dy = np.fft.irfft(train_Q, 502, axis=1)*502, -np.fft.irfft(train_potential, 502, axis=1)*502
test_X, test_dy = np.fft.irfft(test_Q, 502, axis=1)*502, -np.fft.irfft(test_potential, 502, axis=1)*502

# clustering
cluster = KMeans(5, random_state=5)
train_labels = cluster.fit_predict(train_X)
test_labels = cluster.predict(test_X)

label = 1

train_X_set_1, train_y_set_1, train_dy_set_1 = train_X[train_labels==label], train_y[train_labels==label], train_dy[train_labels==label]
test_X_set_1, test_y_set_1, test_dy_set_1 = test_X[test_labels==label], test_y[test_labels==label], test_dy[test_labels==label]

# mean_X = np.mean(train_X_set_1, axis=0)
# Cov = (train_X_set_1 - mean_X).T @ (train_X_set_1 - mean_X) / train_X_set_1.shape[0]
# U, _, _ = np.linalg.svd(Cov)

gamma = np.logspace(-10, 5, 50)
Err_test = []
Err_train = []
for g in gamma:
    kernel = rbfKernel(g)
    model = KernelRidge(kernel, 0)
    model.fit(train_X_set_1, train_y_set_1[:, np.newaxis])
    predict_test_y = model.predict(test_X_set_1)
    predict_train_y = model.predict(train_X_set_1)
    err_test = np.mean((predict_test_y - test_y_set_1)**2)
    err_train = np.mean((predict_train_y - train_y_set_1)**2)
    Err_test.append(err_test)
    Err_train.append(err_train)

import matplotlib.pyplot as plt
plt.plot(gamma, Err_train, 'b', label='train')
plt.plot(gamma, Err_test, 'g', label='test')
plt.semilogx()
plt.xlabel(r'$\gamma$')
plt.ylabel('error')
plt.legend()
plt.show()


# from scipy.special import gamma

# def knn(x, X, n):
#     d = x.shape[1]
#     D = np.sqrt(np.sum((X - x)**2, axis=1))
#     D.sort()
#     r = D[n+1]
#     return (np.pi**(d*0.5))*r**d/gamma(0.5*d+1)

# V = []
# for i in range(1, 20):
#     dens_xt = (train_X_set_1 - mean_X) @ U[:, :i]
#     dens_mean = np.mean(dens_xt, axis=0, keepdims=True)
#     V.append(knn(dens_mean, dens_xt, 30))

# import matplotlib.pyplot as plt
# from matplotlib import ticker
# ax = plt.gca()
# ax.plot(np.arange(1, 20, 1), V, 'bo-')
# ax.set_xlabel('dimension')
# ax.set_ylabel('Volumn')
# ax.semilogy()
# majors = np.arange(1, 25, 5)
# ax.xaxis.set_major_locator(ticker.FixedLocator(majors))
# plt.savefig('curse_dm.png')

# import matplotlib.pyplot as plt
# X = np.linspace(0, 1, 502)
# for i in range(20):
#     plt.plot(X, train_dy_set_1[i])
# plt.show()


# cross-validation
# Gamma = np.logspace(-5, 1, 50)
# Lambda = np.logspace(-14, -3, 50)
# gg, ll = np.meshgrid(Gamma, Lambda)
# Params = np.c_[gg.reshape((-1, 1)), ll.reshape((-1, 1))]
# Error = []

# for gamma, lambda_ in Params:
#     workflow = Workflow(4, gamma, lambda_, rbfKernel, rbfKernel_gd, KernelRidge)
#     kfold = n_split(4, 324, random_state=5)
#     CV = Cross_validation(kfold, workflow)
#     error = CV.run(train_X_set_1, train_y_set_1[:, np.newaxis], train_dy_set_1)
#     Error.append(error)
# Error = np.array(Error).reshape(gg.shape)
# x_min = Error.argmin()//50
# y_min = Error.argmin()%50

# with open('error_surface', 'wb') as f2:
#     pickle.dump(Error, f2)

# import matplotlib.pyplot as plt
# plt.contourf(gg, ll, np.log(Error), 50)
# plt.plot(gg[x_min, y_min], ll[x_min, y_min], 'ko')
# plt.semilogx()
# plt.semilogy()
# plt.show()

# build workflow & performance check
# gamma = 1.0481131341546853e-07
# lambda_ = 4.941713361323838e-12
gamma = 1e-7
lambda_ = 1e-12
# Error_y = []
# Error_dy = []
# for i in range(1, 21):
workflow = Workflow(2, gamma, lambda_, rbfKernel, rbfKernel_gd, KernelRidge)
workflow.fit(train_X_set_1, train_y_set_1[:, np.newaxis])
pred_y, pred_dy = workflow.predict(test_X_set_1)

project_pred_dy = np.sum(pred_dy[:, :, np.newaxis]*workflow.tr_mat_.T, axis=1)
project_test_dy = np.sum((test_dy_set_1 @ workflow.tr_mat_)[:, :, np.newaxis]*workflow.tr_mat_.T, axis=1)

err_y = (pred_y - test_y_set_1)**2
err_dy = np.mean((project_pred_dy - project_test_dy)**2, axis=1)

#     Error_y.append(err_y)
#     Error_dy.append(err_dy)

# Error_y = np.array(Error_y)
# Error_dy = np.array(Error_dy)

# plt.plot(np.arange(1, 21, 1), Error_y, 'bo-')
# plt.plot(np.arange(1, 21, 1), Error_dy, 'go-')
# plt.semilogy()
# plt.show()

import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
n_test = test_X_set_1.shape[0]
ax1.plot(np.arange(0, n_test, 1), err_y, 'b', label='KE error')
ax1.plot(np.arange(0, n_test, 1), err_dy, 'g', label='GD error')
ax1.set_xlabel('test examples')
ax1.set_ylabel('error')
ax1.legend()

X = np.linspace(0, 1, 502)
ax2.plot(X, project_test_dy[51], 'r', label='True')
ax2.plot(X, project_pred_dy[51], 'b', label='Predict')
ax2.set_xlabel('x')
ax2.set_ylabel(r'$\frac{\delta T}{\delta n(x)}$')
ax2.legend()

plt.show()
