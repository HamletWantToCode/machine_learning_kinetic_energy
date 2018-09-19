"""
Functional module:
dim(densG) = (1, f)
dim(alpha) = N
dim(Xi) = (f, N)
"""

import numpy as np 
import pickle

def kinetic_dens(densG, alpha, Xi, gamma):
    nf, N = Xi.shape
    assert densG.shape[1] == nf, ('density should be arranged to (m, nf)')
    assert len(alpha) == N, ('alpha should be arranged to (N)')
    kernel = (gamma*(densG @ Xi))**3
    Ek = kernel @ alpha
    return Ek.real

def kinetic_alpha(alpha, densG, Xi, gamma):
    nf, N = Xi.shape
    assert densG.shape[1] == nf, ('density should be arranged to (m, nf)')
    assert len(alpha) == N, ('alpha should be arranged to (N)')
    kernel = (gamma*(densG @ Xi))**3
    Ek = kernel @ alpha
    return Ek.real

def kinetic_deriv_dens(densG, alpha, Xi, gamma):
    nf, N = Xi.shape
    assert densG.shape[1] == nf, ('density should be arranged to (m, nf)')
    assert len(alpha) == N, ('alpha should be arranged to (N)')
    m = densG.shape[0]
    
    kernel = (densG @ Xi)**2                       # m*f*f*N=m*N
    assert kernel.shape == (m, N)

    dfx_ = np.zeros((m, nf, N))
    for i, ki in enumerate(kernel):
        A = np.repeat(ki.reshape(1, -1), repeats=nf, axis=0)
        assert A.shape == (nf, N)
        dfx_[i] = A*Xi
    return (3*gamma**3)*(dfx_ @ alpha).real      # m*f*N*N = m*f

def kinetic_deriv_alpha(densG, Xi, gamma):
    nf, N = Xi.shape
    assert densG.shape[1] == nf, ('density should be arranged to (m, nf)')
    return (gamma*(densG @ Xi).real)**3   # N

def kinetic_deriv2_dens(densG, alpha, Xi, gamma):
    """
    d2T / dn(q)dn(-q)
    """
    nf, N = Xi.shape
    assert densG.shape[1] == nf, ('density should be arranged to (m, nf)')
    assert len(alpha) == N, ('alpha should be arranged to (N)')
    m = densG.shape[0]

    kernel = densG @ Xi
    assert kernel.shape == (m, N)

    # dn_i(q)dn_i(-q)
    B = np.zeros(((nf+1)//2, N))       # f/2*N
    for i in range((nf+1)//2):
        B[i] = Xi[i]*Xi[-i]
    
    dfx2 = np.zeros((m, (nf+1)//2, N))
    for j, kj in enumerate(kernel):
        A = np.repeat(kj.reshape(1, -1), (nf+1)//2, 0)
        assert A.shape == ((nf+1)//2, N)
        dfx2[j] = A * B
    return 6*(gamma**3)*(dfx2 @ alpha).real     # m*f/2