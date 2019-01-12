import pickle
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

from statslib.main.pca import PrincipalComponentAnalysis
from statslib.main.kernel_ridge import KernelRidge
from statslib.main.pipeline import MyPipe
from statslib.main.utils import rbf_kernel, rbf_kernel_gradient

R = np.random.RandomState(328392)

with open('../example_demo/demo_data', 'rb') as f:
    data = pickle.load(f)
with open('../example_demo/demo_Vx', 'rb') as f1:
    potential = pickle.load(f1)
densx, Ek, dEkx = data[:, 2:], data[:, 1], -potential[:, 1:]
densx_train, densx_test, Ek_train, Ek_test, dEkx_train, dEkx_test = train_test_split(densx, Ek, dEkx, test_size=0.4, random_state=R)

pca = PrincipalComponentAnalysis(50)
C, gamma = 1.09854114e-10, 0.0009102982
# C, gamma = 1e-8, 1e-2
krr = KernelRidge(C=C, gamma=gamma, kernel=rbf_kernel, kernel_gd=rbf_kernel_gradient)
pipe = MyPipe([('reduce_dim', pca), ('regressor', krr)])
pipe.fit(densx_train, Ek_train, dEkx_train)

dEkxt_predict = pipe.predict_gradient(densx_test[0][np.newaxis, :])
dEkxt_test = pipe.named_steps['reduce_dim'].transform_gradient(dEkx_test[0][np.newaxis, :])

plt.plot(np.arange(0, 50, 1), dEkxt_test[0], 'ro-', label='true')
plt.plot(np.arange(0, 50, 1), dEkxt_predict[0], 'bo--', label='predict', fillstyle='none')
plt.xlabel('principal componen')
plt.ylabel(r'$\hat{P}\left[\frac{\delta T}{\delta n(x)}\right]$')
plt.legend()
plt.savefig('../example_demo/projected_derivative.png')