import numpy as np 
from MLEK.main.solver import solver
import matplotlib.pyplot as plt 

np.random.seed(45433)

def data_gen(nbasis, a, b, c):
    X = np.linspace(0, 1, nbasis)
    Vx = -np.sum(a[:, np.newaxis]*np.exp(-0.5*(X[np.newaxis, :]-b[:, np.newaxis])**2/c[:, np.newaxis]**2), axis=0)
    Vq = np.fft.rfft(Vx)/nbasis
    H = np.zeros((nbasis, nbasis), dtype=np.complex128)
    nq = len(Vq)
    for i in range(1, nq):
        np.fill_diagonal(H[i:, :-i], Vq[i].conj())
    return Vq, H

nbasis = 100
nk = 100
mu1 = 10
mu2 = 50
a = np.random.uniform(50, 100, 3)
b = np.random.uniform(0.4, 0.6, 3)
c = np.random.uniform(0.03, 0.1, 3)
Vq, H = data_gen(nbasis, a, b, c)
_, _, dens_q1, En = solver(nk, nbasis, mu1, H)
_, _, dens_q2, _ = solver(nk, nbasis, mu2, H) 
dens_x1 = np.fft.irfft(dens_q1, nbasis)*nbasis
dens_x2 = np.fft.irfft(dens_q2, nbasis)*nbasis
# Vx = np.fft.irfft(Vq, nbasis)*nbasis

kpoints = np.linspace(0, np.pi, nk)
X = np.linspace(0, 1, nbasis)
# plt.plot(X, dens_x1, 'g')
# plt.plot(X, dens_x2, 'b--', alpha=0.7)
for i in range(4):
    plt.plot(kpoints, En[:, i], 'b')
plt.show()
