import pickle
import numpy as np
from GPML.main.optimize import MCMC_Optimize
from GPML.main.gauss_process import Gauss_Process_Regressor
from GPML.main.utils import rbfKernel, rbfKernel_gd_coef

np.random.seed(8)

with open('/Users/hongbinren/Downloads/mnt/project/Burke_paper/quantumX1D', 'rb') as f:
    data = pickle.load(f)
np.random.shuffle(data)
train_data, test_data = data[:1001], data[1001:]
# with open('/Users/hongbinren/Downloads/mnt/project/Burke_paper/potentialX1D', 'rb') as f1:
#     potential = pickle.load(f1)
# nsamples = data.shape[0]
# index = np.arange(0, nsamples, 1, dtype='int')
# np.random.shuffle(index)
# train_data, train_potential = data[index[:40]], potential[index[:40]]
# test_data, test_potential = data[index[60:]], potential[index[60:]]



# N=1
# scaler = 0.06
train_data_single = train_data[train_data[:, 0]==1]
test_data_single = test_data[test_data[:, 0]==1]
train_X, train_y = train_data_single[:100, 2:], train_data_single[:100, 1]
mean_KE = np.mean(train_y)
train_y -= mean_KE

# scaler = np.amax(train_dy, axis=0, keepdims=True)
# np.maximum(1e-10, scaler, out=scaler)
# train_dy /= scaler
# train_y_ = np.r_[train_y, train_dy.reshape(-1)]

test_X, test_y = test_data_single[:, 2:], test_data_single[:, 1]
test_y -= mean_KE
# test_dy /= scaler

# gamma = 0.6
# sigma = 0.25
# Gamma = np.logspace(-4, 1, 30)
# Sigma = np.logspace(-2, 2, 20)
# gg, ss = np.meshgrid(Gamma, Sigma)
# Parameters = np.c_[gg.reshape((-1, 1)), ss.reshape((-1, 1))]
# Likelihood = []
# for gamma, sigma in Parameters:
optimizer = MCMC_Optimize(1e-4, 1e-3, 10000, verbose=1)
# optimizer = GradientAscent(1e-9, 1e-3, 10000, verbose=1)
model = Gauss_Process_Regressor(rbfKernel, rbfKernel_gd_coef, optimizer)
model.fit(train_X, train_y[:, np.newaxis])
y_pred, y_err = model.predict(test_X)

print(np.mean((y_pred - test_y)**2))

import matplotlib.pyplot as plt  
plt.errorbar(test_y, y_pred, yerr=y_err, fmt='o')
plt.plot(test_y, test_y, 'r')
plt.show()



# K_star = np.c_[kernel_gd(test_X, train_X), kernel_hess(test_X, train_X)]
# predict_dy = K_star @ model.coef_
# predict_dy = predict_dy.reshape(test_dy.shape)

# # K_star = kernel_gd(test_X, train_X)
# # predict_dy = K_star @ model.coef_
# # predict_dy = predict_dy.reshape(test_dy.shape)

# import matplotlib.pyplot as plt
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
# ax1.errorbar(test_y, y_pred, yerr=y_err, fmt='o')
# ax1.plot(test_y, test_y, 'r')
# # ax1.set_ylim([-0.5, 1.0])
# X = np.linspace(0, 1, 22)
# ax2.plot(X, test_dy[3], 'r')
# ax2.plot(X, predict_dy[3], 'b--', alpha=0.5)
# plt.show()
