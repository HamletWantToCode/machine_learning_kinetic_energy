# quantum solver 

import numpy as np 
from scipy.linalg import eig_banded

def solver(nk, nbasis, dmu, Vq):
    kpoints = np.linspace(0, np.pi, nk)
    # build and solve eigenvalue problem
    max_pos_freq = len(Vq)
    T = 0
    density = np.zeros(nbasis, dtype=np.complex64)
    for k in kpoints:
        hamiton_mat = np.zeros((max_pos_freq, nbasis), dtype=np.complex64)    
        for i in range(max_pos_freq):
            for j in range(i, nbasis):
                if i==0:
                    hamiton_mat[max_pos_freq-1, j] = 0.5*(k+(j-nbasis//2)*2*np.pi)**2 + Vq[0]
                else:
                    hamiton_mat[max_pos_freq-i-1, j] = Vq[i]
        En_k, Uq_k = eig_banded(hamiton_mat, overwrite_a_band=True, select='a')
        if k == 0:
            mu = En_k[0] + dmu

        # compute electron density
        # compute kinetic energy
        num_mat_eigspace = np.zeros((nbasis, nbasis))
        for i in range(nbasis):
            if En_k[i] > mu:
                break
            num_mat_eigspace[i, i] = 1
        density_mat_kspace = Uq_k @ (num_mat_eigspace @ (Uq_k.T).conj())

        density_k = np.zeros(nbasis, dtype=np.complex64)
        T_k = 0
        for i in range(nbasis):
            density_k[i] = np.trace(density_mat_kspace, offset=i)
            T_k += 0.5*((k+(i-nbasis//2)*2*np.pi)**2)*(density_mat_kspace[i, i]).real
        T += T_k
        density += density_k
    return T/nk, density/nk #, En_k




