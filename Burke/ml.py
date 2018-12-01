import pickle
import numpy as np
from statslib.main.kernel_ridge import KernelRidge
from statslib.tools.utils import rbfKernel, n_split, meanSquareError
from statslib.main.cross_validation import Cross_validation

np.random.seed(8)

with open('/Users/hongbinren/Downloads/mnt/project/Burke_paper/quantumX1D', 'rb') as f:
    data = pickle.load(f)
np.random.shuffle(data)
train_data = data[:1001]
test_data = data[1001:]

# N=1
train_n = train_data[train_data[:, 0]==1]
test_n = test_data[test_data[:, 0]==1]
train_X, train_y = train_n[:100, 2:], train_n[:100, 1]
mean_KE = np.mean(train_y)
train_y /= mean_KE
test_X, test_y = test_n[:, 2:], test_n[:, 1]
test_y /= mean_KE

# plot coef contour
Sigma = np.logspace(-3, 4, 200)
Lambda = np.logspace(-16, -4, 100)
xx, yy = np.meshgrid(Sigma, Lambda)
XY = np.c_[xx.reshape((-1, 1)), yy.reshape((-1, 1))]
Z = []
for (sigma, lambda_) in XY:
    gamma = 1.0/(2*sigma**2)
    kernel = rbfKernel(gamma)
    model = KernelRidge(kernel, lambda_)
    sp = n_split(5, 100, 53)
    CV = Cross_validation(sp, model, meanSquareError)
    avg_err = CV.run(train_X, train_y[:, np.newaxis])
    Z.append(avg_err)
zz = np.array(Z).reshape(xx.shape)
surf_data = np.vstack([xx, yy, zz])

with open('error_surf', 'wb') as f:
    pickle.dump(surf_data, f)

# import matplotlib.pyplot as plt
# plt.contourf(xx, yy, zz)
# plt.colorbar()
# plt.semilogx()
# plt.semilogy()
# plt.show()

