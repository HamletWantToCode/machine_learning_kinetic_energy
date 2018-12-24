import pickle
import numpy as np
from statslib.main.workflow import Workflow
from statslib.main.cross_validation import Cross_validation
from statslib.main.kernel_ridge import KernelRidge
from statslib.tools.utils import n_split, rbfKernel, rbfKernel_gd

np.random.seed(83223)

with open('quantumX1D', 'rb') as f:
    data = pickle.load(f)
with open('potentialX1D', 'rb') as f1:
    potential = pickle.load(f1)
ne = 1
dens_X, Ek, dEk = data[data[:, 0]==ne, 2:], data[data[:, 0]==ne, 1], -potential[data[:, 0]==ne, 1:]
index = np.arange(0, dens_X.shape[0], 1, 'int')
np.random.shuffle(index)

train_X, train_y, train_dy = dens_X[index[:250]], Ek[index[:250]], dEk[index[:250]]
test_X, test_y, test_dy = dens_X[index[250:]], Ek[index[250:]], dEk[index[250:]]

gamma = np.logspace(-5, -1, 50)
lambda_ = np.logspace(-14, -5, 50)
gg, ll = np.meshgrid(gamma, lambda_)
Parameters = np.c_[gg.reshape((-1, 1)), ll.reshape((-1, 1))]
Error = []
for g, l in Parameters:
    workflow = Workflow(4, g, l, rbfKernel, rbfKernel_gd, KernelRidge)
    sp = n_split(5, 250, random_state=6567)
    cv = Cross_validation(sp, workflow)
    avgerr = cv.run(train_X, train_y[:, np.newaxis], train_dy)
    Error.append(avgerr)
Error_surf = np.array(Error).reshape(gg.shape)

with open('error_surf', 'wb') as f2:
    pickle.dump(Error_surf, f2)

x_min, y_min = Error_surf.argmin()//50, Error_surf.argmin()%50
print(gg[x_min, y_min], ll[x_min, y_min])

import matplotlib.pyplot as plt
plt.contourf(gg, ll, np.log(Error_surf), 50)
plt.xlabel(r'$\gamma$')
plt.ylabel(r'$\lambda$')
plt.plot(gg[x_min, y_min], ll[x_min, y_min], 'ko')
plt.colorbar()
plt.semilogx()
plt.semilogy()
plt.show()
