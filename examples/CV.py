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
data_q, potential_q = data[:, 1:], potential
data_q_2, potential_q_2 = data_q[data[:, 1].real > 1.8], potential_q[data[:, 1] > 1.8]
data_X, potential_X = np.fft.irfft(data_q_2, 100, axis=1)*100, np.fft.irfft(potential_q_2, 100, axis=1)*100
index = np.arange(0, data_X.shape[0], 1, 'int')
np.random.shuffle(index)
train_X, train_y, train_dy = data_X[index[:500]], data[index[:500], 0].real, -potential_X[index[:500]]
test_X, test_y, test_dy = data_X[index[500:]], data[index[500:], 0].real, -potential_X[index[500:]]

# Gamma, Lambda = np.logspace(-10, -3, 20), np.logspace(-10, -1, 20)
# gg, ll = np.meshgrid(Gamma, Lambda)
# Params = np.c_[gg.reshape((-1, 1)), ll.reshape((-1, 1))]
# Error = []
# for g, l in Params:
g = 1e-4
l = 1e-8
workflow = Workflow(2, g, l, rbfKernel, rbfKernel_gd, KernelRidge)
#     kfold = n_split(10, 500, random_state=5)
#     CV = Cross_validation(kfold, workflow)
#     error = CV.run(train_X, train_y[:, np.newaxis], train_dy)
#     Error.append(error)
# Error = np.array(Error).reshape(gg.shape)
workflow.fit(train_X, train_y[:, np.newaxis])
pred_y, pred_dy = workflow.predict(test_X)

project_pred_dy = np.sum(pred_dy[:, :, np.newaxis]*workflow.tr_mat_.T, axis=1)
project_test_dy = np.sum((test_dy @ workflow.tr_mat_)[:, :, np.newaxis]*workflow.tr_mat_.T, axis=1)

err_y = (pred_y - test_y)**2
err_dy = np.mean((project_pred_dy - project_test_dy)**2, axis=1)

import matplotlib.pyplot as plt  
n_test = test_dy.shape[0]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
# ax1.plot(np.arange(0, n_test, 1), err_y, 'b')
ax1.plot(np.arange(0, n_test, 1), err_dy, 'g')

X = np.linspace(0, 1, 100)
ax2.plot(X, project_test_dy[34], 'r')
ax2.plot(X, project_pred_dy[34], 'b')

plt.show()

# import pickle
# with open('data_file/error_surf', 'wb') as f1:
#     pickle.dump(Error, f1)

# x_min, y_min = Error.argmin()//20, Error.argmin()%20
# print(gg[x_min, y_min])
# print(ll[x_min, y_min])
# print(Error[x_min, y_min])

# import matplotlib.pyplot as plt 
# plt.contourf(gg, ll, np.log(Error), 40)
# plt.plot(gg[x_min, y_min], ll[x_min, y_min], 'ko')
# plt.xlabel(r'$\gamma$')
# plt.ylabel(r'$\lambda$')
# plt.semilogx()
# plt.semilogy()
# plt.colorbar()
# plt.savefig('data_file/error_surf.png')

    

