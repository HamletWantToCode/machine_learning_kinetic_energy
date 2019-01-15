# test solver

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from MLEK.main.solver import solver
from MLEK.main.utils import simple_potential_gen
from MLEK.tools.plot_tools import plot_real_space_density

nk = 100
nbasis = 10
low_V0, high_V0 = 5, 8
low_Phi0, high_Phi0 = -0.1, 0.1
params_gen = simple_potential_gen(nbasis, low_V0, high_V0, low_Phi0, high_Phi0, 38782)

fig = plt.figure()
ax = fig.gca()
X = np.linspace(0, 1, 100)
hamilton_mat, Vq = next(params_gen)
T, densq, mu, En = solver(nk, nbasis, hamilton_mat, debug=True)

k_points = np.linspace(0, np.pi, 100)
for i in range(4):
    plt.plot(k_points, En.T[i], 'b')
plt.savefig('../periodic/energy_band.png')


# DENSX = np.zeros((20, 100))
# for i in range(20):
#     hamilton_mat, Vq = next(params_gen)
#     T, densq, mu = solver(nk, nbasis, hamilton_mat)
#     densx = np.fft.irfft(densq, 100)*100
#     DENSX[i] = densx
# plot_real_space_density(DENSX, out_dir='test_demo')



# def V_gen(nbasis, V0):
#     hamilton_mat = np.zeros((nbasis, nbasis), dtype=np.complex64)
#     np.fill_diagonal(hamilton_mat[1:, :-1], V0*(-0.25))
#     Vq = np.zeros(nbasis, dtype=np.complex64)
#     Vq[0], Vq[1] = -0.5*V0, -0.25*V0
#     return hamilton_mat, Vq

# omega = 2*np.pi*np.sqrt(V0)
# f = lambda x: np.sqrt(omega/np.pi)*np.exp(-omega*x**2)
# y = f(X)

# print(delta_En_k)
# # print(Vq[1])
# print(dens_q[0])
# print((dens_q[0]**3)*(np.pi**2)/6)
# print(T)
