# gauss process fitting
import pickle
import numpy as np 
from GPML.main.gauss_process import Gauss_Process_Regressor
from GPML.main.utils import rbfKernel, rbfKernel_gd, rbfKernel_hess

np.random.rand(8)

with open('/Users/hongbinren/Documents/program/MLEK/data_file/quantum', 'rb') as f:
    data = pickle.load(f)
with open('/Users/hongbinren/Documents/program/MLEK/data_file/potential', 'rb') as f1:
    potential = pickle.load(f1)
n_example = data.shape[0]
index = np.arange(0, n_example, 1, 'int')
np.random.shuffle(index)
train_data, train_potential = data[index[:101]], potential[index[:101]]
test_data, test_potential = data[index[101:]], potential[index[101:]]

# train-test feature/target
train_X, train_y, train_dy = np.fft.irfft(train_data[:, 1:], 100, axis=1)*100, train_data[:, 0].real, -np.fft.irfft(train_potential, 100, axis=1)*100
test_X, test_y, test_dy = np.fft.irfft(test_data[:, 1:], 100, axis=1)*100, test_data[:, 0].real, -np.fft.irfft(test_potential, 100, axis=1)*100 

# compress the data
n_train = train_X.shape[0]
Mean_X = np.mean(train_X, axis=0)
Mean_dy = np.mean(train_dy, axis=0)
Cov = (train_X - Mean_X).T @ (train_X - Mean_X) / n_train
U, S, _ = np.linalg.svd(Cov)
rank = 2
train_X_t = (train_X - Mean_X) @ U[:, :rank]
train_dy_t = (train_dy - Mean_dy) @ U[:, :rank]
## standardize train
mean_dy_t = np.mean(train_dy_t, axis=0)
var_dy_t = np.sqrt(np.var(train_dy_t, axis=0))
train_dy_std = (train_dy_t - mean_dy_t) / var_dy_t
project_train_dy = np.sum(train_dy_t[:, :, np.newaxis]*(U[:, :rank]).T, axis=1) + Mean_dy

Mean_y = np.mean(train_y)
var_y = np.sqrt(np.var(train_y))
train_y_std = (train_y - Mean_y) / var_y

train_target = np.r_[train_y_std, train_dy_std.reshape(-1)]

# standardize test
test_X_t = (test_X - Mean_X) @ U[:, :rank]
test_y_std = (test_y - Mean_y) / var_y
test_dy_t = (test_dy - Mean_dy) @ U[:, :rank]
test_dy_std = (test_dy_t - mean_dy_t) / var_dy_t
project_test_dy = np.sum(test_dy_t[:, :, np.newaxis]*(U[:, :rank]).T, axis=1) + Mean_dy

# training performance on y and dy
gamma = 0.1
sigma = 1e-5
# optimizer = MCSA_Optimize(0.1, 2000, 1e-4, 10000, verbose=0)
model = Gauss_Process_Regressor(rbfKernel, None, gradient_on=True, kernel_gd=rbfKernel_gd, kernel_hess=rbfKernel_hess)
model.fit(train_X_t, train_target[:, np.newaxis], optimize_on=False, gamma=gamma, sigma=sigma)
y_pred, y_err = model.predict(test_X_t)
print(model.gamma_, model.sigma_)

predict_KE = y_pred*var_y + Mean_y
err = (predict_KE - test_y)**2

gamma = model.gamma_
K_star = np.c_[rbfKernel_gd(gamma, test_X_t, train_X_t), rbfKernel_hess(gamma, test_X_t, train_X_t)]
predict_project_dy = (K_star @ model.coef_).reshape(test_dy_t.shape)
predict_project_dy *= var_dy_t
predict_project_dy += mean_dy_t
predict_dy = np.sum(predict_project_dy[:, :, np.newaxis]*(U[:, :rank]).T, axis=1) + Mean_dy
err_dy = np.mean((predict_dy - project_test_dy)**2, axis=1)

import matplotlib.pyplot as plt
n_test = test_X.shape[0]
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
ax1.plot(np.arange(0, n_test, 1), err, 'b', label='KE')
ax1.plot(np.arange(0, n_test, 1), err_dy, 'g', label='project gd')
ax1.set_ylim([0, 5])
ax1.legend()

X = np.linspace(0, 1, 100)
ax2.plot(X, project_test_dy[40], 'r', label='project gd')
ax2.plot(X, predict_dy[40], 'b', label='predict gd', alpha=0.5)
ax2.legend()
plt.show()
