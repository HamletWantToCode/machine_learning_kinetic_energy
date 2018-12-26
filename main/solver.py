# quantum solver

import numpy as np
from scipy.linalg import eigh

def solver(nk, nbasis, mu, hamiton_mat):
    kpoints = np.linspace(0, np.pi, nk)
    # build and solve eigenvalue problem
    T = 0
    density = np.zeros(nbasis, dtype=np.complex64)
    for ki, k in enumerate(kpoints):
        kinetic_term = np.array([0.5*(k+(i-nbasis//2)*2*np.pi)**2 for i in range(nbasis)])
        np.fill_diagonal(hamiton_mat, kinetic_term)
        En_k, Uq_k = eigh(hamiton_mat, overwrite_a=True, overwrite_b=True)
        # compute mu
        if k == 0:
            b = En_k[0]         # set the minimum of band energy to 0 !
        # compute electron density
        # compute kinetic energy
        num_mat_eigspace = np.zeros((nbasis, nbasis))
        for i in range(nbasis):
            if En_k[i] <= mu + b:
                num_mat_eigspace[i, i] = 1
            else:
                break
        density_mat_kspace = Uq_k @ (num_mat_eigspace @ (Uq_k.T).conj())

        density_k = np.zeros(nbasis, dtype=np.complex64)
        T_k = 0
        for i in range(nbasis):
            density_k[i] = np.trace(density_mat_kspace, offset=i)
            T_k += 0.5*((k+(i-nbasis//2)*2*np.pi)**2)*(density_mat_kspace[i, i]).real
        T += T_k
        density += density_k
    return T/nk, density/nk



