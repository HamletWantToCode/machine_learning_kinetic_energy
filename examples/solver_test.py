# test solver

import numpy as np
from MLEK.main.solver import solver
from MLEK.main.utils import irfft

def V_gen(nbasis, V0):
    hamilton_mat = np.zeros((nbasis, nbasis), dtype=np.complex64)
    np.fill_diagonal(hamilton_mat[1:, :-1], V0*(-0.25))
    Vq = np.zeros(nbasis, dtype=np.complex64)
    Vq[1] = -0.25*V0
    return hamilton_mat, Vq 

nk = 100
nbasis = 10
V0 = 100
hamilton_mat, Vq = V_gen(nbasis, V0)
dmu = 10
T, mu, dens_q = solver(nk, nbasis, dmu, hamilton_mat)

print(dens_q[0])
X = np.linspace(0, 1, 100)
# dens_x = np.fft.irfft(dens_q, 100)*100
# X = np.r_[np.linspace(0, 0.5, 51), np.linspace(-0.5, 0, 49)] 
dens_x = irfft(dens_q, 100)

# print(delta_En_k)
# # print(Vq[1])
# print(dens_q[0])
# print((dens_q[0]**3)*(np.pi**2)/6)
# print(T)

omega = 2*np.pi*np.sqrt(V0)
f = lambda x: np.sqrt(omega/np.pi)*np.exp(-omega*x**2)
y = f(X)

import matplotlib.pyplot as plt

plt.plot(X, dens_x, 'b')
plt.plot(X, y, 'r')
plt.show()
