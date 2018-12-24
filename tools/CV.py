import pickle
import numpy as np 
import matplotlib.pyplot as plt
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
dens_X, Ek, Vx = np.fft.irfft(data[:, 1:], 100, axis=1)*100, data[:, 0].real, np.fft.irfft(Vq, 100, axis=1)*100
index = np.arange(0, n, 1, 'int')
np.random.shuffle(index)
train_X, train_y, train_dy = dens_X[index[:500]], Ek[index[:500]], -Vx[index[:500]]
test_X, test_y, test_dy = dens_X[index[500:]], Ek[index[500:]], -Vx[index[500:]]

# gamma = 0.002811768697974228
# lambda_ = 4.291934260128778e-06

gamma = np.logspace(-5, -1, 50)
lambda_ = np.logspace(-14, -5, 50)
gg, ll = np.meshgrid(gamma, lambda_)
Parameters = np.c_[gg.reshape((-1,1)), ll.reshape((-1,1))]
Error = []
for g, l in Parameters:
    workflow = Workflow(n_components=2, gamma=g, lambda_=l, kernel=rbfKernel, kernel_gd=rbfKernel_gd, model=KernelRidge)
    sp = n_split(5, 500, random_state=6567)
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

