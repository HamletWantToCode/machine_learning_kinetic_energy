# test solver

import numpy as np
from MLEK.main.solver import solver
from MLEK.main.utils import irfft

# def V_kspace(V0, max_k):
#     Vq = np.zeros(max_k, dtype=np.complex64)
#     for i in range(1, max_k):
#         theta = np.random.uniform(0, 2*np.pi)
#         Vq[i] = -V0*(np.cos(theta) + 1j*np.sin(theta))
#     return Vq

def V_kspace(V0):
    Vq = np.zeros(2)
    Vq[0], Vq[1] = -0.5, -0.25
    Vq *= V0
    return Vq

nk = 100
nbasis = 100
dmu = 20
Vq = V_kspace(400)
T, dens_q = solver(nk, nbasis, dmu, Vq)

X = np.linspace(0, 1, 100)
# dens_x = np.fft.irfft(dens_q, 100)*100
# X = np.r_[np.linspace(0, 0.5, 51), np.linspace(-0.5, 0, 49)] 
dens_x = irfft(dens_q, 100)

# print(En[-1, 1] - En[-1, 0])
# print(Vq[1])
# print(dens[0])
# print((dens[0]**3)*(np.pi**2)/6)
# print(T)

omega = 2*np.pi*20
f = lambda x: np.sqrt(omega/np.pi)*np.exp(-omega*x**2)
y = f(X)

import matplotlib.pyplot as plt

plt.plot(X, dens_x, 'b')
plt.plot(X, y, 'r')
plt.show()
