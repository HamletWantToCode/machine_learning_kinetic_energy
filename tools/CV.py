import pickle
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from statslib.main.kernel_ridge import KernelRidge
from statslib.main.workflow import Workflow
from statslib.tools.utils import rbfKernel, rbfKernel_gd, n_split
from statslib.main.cross_validation import Cross_validation

np.random.seed(5940232)

with open('../data_file/quantum', 'rb') as f:
    data = pickle.load(f)
with open('../data_file/potential', 'rb') as f1:
    Vq = pickle.load(f1)
n = data.shape[0]
dens_X, Ek, mu_reduced, Vx = np.fft.irfft(data[:, 1:], 100, axis=1)*100, data[:, 0].real, Vq[:, 0].real, np.fft.irfft(Vq[:, 1:], 100, axis=1)*100
index = np.arange(0, n, 1, 'int')
np.random.shuffle(index)
train_X, train_y, train_mu, train_dy = dens_X[index[:500]], Ek[index[:500]], mu_reduced[index[:500]], -Vx[index[:500]]
test_X, test_y, test_mu, test_dy = dens_X[index[500:]], Ek[index[500:]], mu_reduced[index[500:]], -Vx[index[500:]]

gamma = 1e-3
lambda_ = 1e-10

workflow = Workflow(n_components=2, gamma=gamma, lambda_=lambda_, kernel=rbfKernel, kernel_gd=rbfKernel_gd, model=KernelRidge)
workflow.fit(train_X, train_y[:, np.newaxis])
pred_y, pred_dyt = workflow.predict(test_X)
test_dyt = test_dy @ workflow.tr_mat_

project_test_dy = np.sum(test_dyt[:, :, np.newaxis]*workflow.tr_mat_.T, axis=1)
project_pred_dy = np.sum(pred_dyt[:, :, np.newaxis]*workflow.tr_mat_.T, axis=1)

pred_mu = np.mean(project_pred_dy - project_test_dy, axis=1)

err_dy = np.mean((pred_dyt - test_dyt)**2, axis=1)
ii = err_dy.argmax()
err_y = (pred_y - test_y)**2

X = np.linspace(0, 1, 100)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(X, project_pred_dy[ii], 'b')
ax1.plot(X, project_test_dy[ii], 'r')

ax2.plot(test_y, pred_y, 'bo')
ax2.plot(test_y, test_y, 'r')
plt.show()


# gamma = np.logspace(-8, -1, 50)
# lambda_ = np.logspace(-14, -5, 50)
# gg, ll = np.meshgrid(gamma, lambda_)
# Parameters = np.c_[gg.reshape((-1,1)), ll.reshape((-1,1))]
# Error = []
# for g, l in Parameters:
#     workflow = Workflow(n_components=1, gamma=g, lambda_=l, kernel=rbfKernel, kernel_gd=rbfKernel_gd, model=KernelRidge)
#     sp = n_split(5, 500, random_state=6567)
#     cv = Cross_validation(sp, workflow)
#     avgerr = cv.run(train_X, train_y[:, np.newaxis], train_dy)
#     Error.append(avgerr)
# Error_surf = np.array(Error).reshape(gg.shape)
# 
# with open('error_surf', 'wb') as f2:
#     pickle.dump(Error_surf, f2)
# 
# x_min, y_min = Error_surf.argmin()//50, Error_surf.argmin()%50
# print(gg[x_min, y_min], ll[x_min, y_min])
# 
# import matplotlib.pyplot as plt
# plt.contourf(gg, ll, np.log(Error_surf), 50)
# plt.xlabel(r'$\gamma$')
# plt.ylabel(r'$\lambda$')
# plt.plot(gg[x_min, y_min], ll[x_min, y_min], 'ko')
# plt.colorbar()
# plt.semilogx()
# plt.semilogy()
# plt.show()
# 
