# test solver

import numpy as np
from MLEK.main.solver import solver
from MLEK.main.utils import irfft

def V_gen(nbasis, V0, max_k):
    hamilton_mat = np.zeros((nbasis, nbasis), dtype=np.complex64)
    for i in range(1, max_k):
        theta = np.random.uniform(0, 2*np.pi)
        Vq_conj = -V0*(np.cos(theta) - 1j*np.sin(theta))
        np.fill_diagonal(hamilton_mat[i:, :-i], Vq_conj)
    return hamilton_mat

nk = 100
nbasis = 100
V0 = 10
hamilton_mat = V_gen(nbasis, V0, 50)
dmu = 20
T, dens_q, delta_En_k = solver(nk, nbasis, dmu, hamilton_mat)

# X = np.linspace(0, 1, 100)
# dens_x = np.fft.irfft(dens_q, 100)*100
# X = np.r_[np.linspace(0, 0.5, 51), np.linspace(-0.5, 0, 49)] 
# dens_x = irfft(dens_q, 100)

# print(delta_En_k)
# # print(Vq[1])
# print(dens_q[0])
# print((dens_q[0]**3)*(np.pi**2)/6)
# print(T)

# omega = 2*np.pi*20
# f = lambda x: np.sqrt(omega/np.pi)*np.exp(-omega*x**2)
# y = f(X)

# import matplotlib.pyplot as plt

# plt.plot(X, dens_x, 'b')
# plt.plot(X, y, 'r')
# plt.show()
