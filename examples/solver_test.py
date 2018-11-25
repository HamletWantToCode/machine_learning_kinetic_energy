# test solver

import numpy as np
from MLEK.main.solver import solver

def V_kspace(V0, max_k):
    Vq = np.zeros(max_k, dtype=np.complex64)
    for i in range(1, max_k):
        theta = np.random.uniform(0, 2*np.pi)
        Vq[i] = -V0*(np.cos(theta) + 1j*np.sin(theta))
    return Vq

nk = 100
nbasis = 100
mu = 40
Vq = V_kspace(10, 50)
# kpoints = np.linspace(0, np.pi, nk)

# T = 0
# dens = 0
# # En = np.zeros((nk, nbasis))
# for i, k in enumerate(kpoints):
#     T_k, dens_k = solver(k, nbasis, mu, Vq)
#     T += T_k
#     dens += dens_k
# #     En[i] = En_k
# dens/=nk
# T/=nk
T, dens = solver(nk, nbasis, mu, Vq)

# print(En[-1, 1] - En[-1, 0])
print(Vq[1])
print(dens[0])
print((dens[0]**3)*(np.pi**2)/6)
print(T)

# import matplotlib.pyplot as plt
# for i in range(4):
#     plt.plot(kpoints, En[:, i], 'b')
# plt.hlines(mu, 0, np.pi, 'r', linestyles='--')
# plt.show()
