# test solver 

import numpy as np 
from MLEK.main.solver import solver

def V_kspace(V0):
    Vq = np.zeros(2)
    Vq[0], Vq[1] = -0.5, -0.25
    return V0*Vq

nk = 100
nbasis = 31
mu = 40
Vq = V_kspace(0)
kpoints = np.linspace(0, np.pi, nk)

T = 0
dens = 0
En = np.zeros((nk, nbasis))
for i, k in enumerate(kpoints):
    T_k, dens_k, En_k = solver(k, nbasis, mu, Vq)
    T += T_k
    dens += dens_k
    En[i] = En_k
dens/=nk
T/=nk

print(En[-1, 1] - En[-1, 0])
print(Vq[1])
print(dens[0])
print((dens[0]**3)*(np.pi**2)/6)
print(T)

import matplotlib.pyplot as plt 
for i in range(4):
    plt.plot(kpoints, En[:, i], 'b')
plt.hlines(mu, 0, np.pi, 'r', linestyles='--')
plt.show()
