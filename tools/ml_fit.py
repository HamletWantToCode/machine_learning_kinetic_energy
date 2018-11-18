import pickle
import numpy as np 
from ..main.gauss_kernel_fitting import NewLoss_KRR, Special_SGD
from statslib.tools.utils import rbfKernel, meanSquareError

# from sklearn.preprocessing import StandardScaler

np.random.seed(5)

fname = '/Users/hongbinren/Downloads/mnt/project/ML_periodic/quantum1D'
with open(fname, 'rb') as f:
    fst_principal = pickle.load(f)
fname1 = '/Users/hongbinren/Downloads/mnt/project/ML_periodic/potential1D'
with open(fname1, 'rb') as f1:
    potential = pickle.load(f1)
n_samples = fst_principal.shape[0]
index = np.arange(0, n_samples, 1, dtype=int)
np.random.shuffle(index)
# scaler = StandardScaler()
train_index = index[:300]
test_index = index[350:450]
train_density, train_kinetic_energy = fst_principal[train_index, :-1], fst_principal[train_index, -1]
# std_train_density = scaler.fit_transform(train_density)
# train_density_x = np.fft.ifft(train_density[:50], axis=1)*train_density.shape[1]
# X = np.linspace(0,1, train_density.shape[1])
# import matplotlib.pyplot as plt 
# for item in train_density_x:
#     plt.plot(X, item)
# plt.show()


mean = np.mean(train_kinetic_energy)
centered_train_kinetic_energy = train_kinetic_energy - mean

train_external_potential = potential[train_index, 1:]                        
# the fourier components of potential only 
# preserve positive frequency part, negative
# part is the same, and we regenerate it from the following code
train_external_potential = np.c_[train_external_potential, train_external_potential[:, ::-1][:, :-1]]    
train_chemical_potential = np.zeros_like(train_external_potential)
train_chemical_potential[:, 0] = potential[train_index, 0]
train_total_potential = train_external_potential - train_chemical_potential

test_density, test_kinetic_energy = fst_principal[test_index, :-1], fst_principal[test_index, -1]
test_external_potential = potential[test_index, 1:]
test_external_potential = np.c_[test_external_potential, test_external_potential[:, ::-1][:, :-1]]    
test_chemical_potential = np.zeros_like(test_external_potential)
test_chemical_potential[:, 0] = potential[test_index, 0]
test_total_potential = test_external_potential - test_chemical_potential
# std_test_density = scaler.transform(test_density)
centered_test_kinetic_energy = test_kinetic_energy - mean

gamma = 0.1
rbf = rbfKernel(gamma)
from statslib.main.kernel_ridge import KernelRidge
model0 = KernelRidge(rbf, 1e-5)
model0.fit(train_density, centered_train_kinetic_energy)
# predict_kinetic_energy = model.predict(test_density)
alpha0 = model0.coef_
gd = Special_SGD(1e-3, 1e-3, 2000, alpha0=alpha0, n_batch=10, verbose=1)
model = NewLoss_KRR(rbf, 1e-5, 1, gd)
model.fit(train_density, centered_train_kinetic_energy, -train_total_potential, gamma)
predict_kinetic_energy = model.predict(test_density)
err = meanSquareError(predict_kinetic_energy, centered_test_kinetic_energy)
print(err)
# import matplotlib.pyplot as plt 
# plt.plot(centered_test_kinetic_energy, predict_kinetic_energy, 'bo')
# plt.plot(centered_test_kinetic_energy, centered_test_kinetic_energy, 'r')
# plt.show()


test_example = test_density[5]
test_example_potential = test_total_potential[5] 
KM = rbf(test_example[np.newaxis, :], train_density)
alpha = model.coef_
delta_x = test_density[5][np.newaxis, :] - model.X_fit_
test_example_gradient = -2*gamma*((alpha * KM) @ delta_x)
test_example_gradient_0 = -2*gamma*((alpha0 * KM) @ delta_x)


import matplotlib.pyplot as plt 
n = len(test_example_potential)
X = np.linspace(0, 1, n)
test_example_Vx = -np.fft.ifft(test_example_gradient[0])*n
test_example_Vx_std = np.fft.ifft(test_example_potential)*n
test_example_Vx_0 = -np.fft.ifft(test_example_gradient_0[0])*n
plt.plot(X, test_example_Vx_std, 'r')
plt.plot(X, test_example_Vx, 'b-')
plt.plot(X, test_example_Vx_0, 'g-')
plt.show()

