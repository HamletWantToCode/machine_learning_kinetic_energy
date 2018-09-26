import numpy as np
from numba import jit
import warnings
warnings.filterwarnings(action='ignore')

@jit(nopython=True)
def hamilton_operator_G_1D(kx, nG, vext, lambda0):
    H = np.zeros((nG, nG), np.complex64)
    for i in range(nG):
        H[i, i] = 0.5*lambda0*(kx + 2*np.pi*(i-(nG-1)//2))**2 + vext[0]
        for j in range(i+1, nG):
            if (j-i)<=(nG-1)//2:
                H[j, i], H[i, j] = vext[j-i], np.conjugate(vext[j-i])
    return H

@jit(nopython=True)
def hamilton_operator_G_2D(kx, ky, nG, vext, lambda0):
    H_ = np.zeros((nG, nG, nG, nG), np.complex64)
    for i in range(nG**2):
        m_, n_ = i//nG, i%nG
        H_[m_, n_, m_, n_] = 0.5*lambda0*((kx + 2*np.pi*(m_-(nG-1)//2))**2 + (ky + 2*np.pi*(n_-(nG-1)//2))**2) + vext[0, 0]
        for j in range(i+1, nG**2):
            m1_, n1_ = j//nG, j%nG
            if m1_-m_<=(nG-1)//2 and abs(n1_-n_)<=(nG-1)//2:
                if n1_==n_:
                    H_[m1_, n1_, m_, n1_] = vext[m1_-m_, 0]
                    H_[m_, n1_, m1_, n1_] = vext[m_-m1_, 0]
                if n1_>n_:
                    H_[m1_, n1_, m_, n_], H_[m1_, n_, m_, n1_] = vext[m1_-m_, n1_-n_], np.conjugate(vext[m1_-m_, n1_-n_])
                    H_[m_, n1_, m1_, n_], H_[m_, n_, m1_, n1_] = vext[m_-m1_, n1_-n_], np.conjugate(vext[m_-m1_, n1_-n_])
    H = H_.reshape((nG**2, nG**2))
    return H

def solver1D(kpoints, nG, vext, p_ix, queue1, queue2):
    m = len(kpoints)
    part_band, part_uG = np.zeros((m, nG)), np.zeros((m, nG, nG), np.complex64)
    for i, kx in enumerate(kpoints):
        H = hamilton_operator_G_1D(kx, nG, vext, 1)
        eigval, eigvec = np.linalg.eigh(H)
        part_band[i], part_uG[i] = eigval, eigvec
    return queue1.put((p_ix, part_band)), queue2.put((p_ix, part_uG))

def solver2D(kpoints, nG, vext):
    m = len(kpoints)
    part_band, part_uG = np.zeros((m, nG**2)), np.zeros((m, nG**2, nG**2), np.complex64)
    for i, (kx, ky) in enumerate(kpoints):
        H = hamilton_operator_G_2D(kx, ky, nG, vext, 1)
        eigval, eigvec = np.linalg.eigh(H)
        part_band[i], part_uG[i] = eigval, eigvec
    return part_band, part_uG
    
                






