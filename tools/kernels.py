# heterogeneous kernel

import numpy as np 

def se_kernel(Gamma):
    def se_func(X, Y=None):
        if Y is None:
            Y = X
        (N1, D), N = X.shape, Y.shape[0]
        diff = X[:, :, np.newaxis] - Y.T
        dist = ((diff**2)/Gamma[np.newaxis, :, np.newaxis]).sum(axis=1)
        K = np.exp(-0.5*dist)
        return K
    return se_func

def se_kernel_gd(Gamma):
    def se_gd(X, Y=None):
        if Y is None:
            Y = X
        (N1, D), N = X.shape, Y.shape[0]
        diff = X[:, :, np.newaxis] - Y.T
        dist = ((diff**2)/Gamma[np.newaxis, :, np.newaxis]).sum(axis=1)
        K = np.exp(-0.5*dist)
        K_gd = -(diff/Gamma[np.newaxis, :, np.newaxis])*K[:, np.newaxis, :]
        K_gd = K_gd.reshape((N1*D, N))
        return K_gd
    return se_gd

def se_kernel_hess(Gamma):
    def se_hess(X, Y=None):
        if Y is None:
            Y = X
        (N1, D), N = X.shape, Y.shape[0]
        diff = X[:, :, np.newaxis] - Y.T
        dist = ((diff**2)/Gamma[np.newaxis, :, np.newaxis]).sum(axis=1)
        K = np.exp(-0.5*dist) 
        K_hess = np.zeros((N1*D, N*D))
        E = np.eye(D)
        for n in range(0, N1*D, D):
            n_, i = n//D, n%D
            for m in range(0, N*D, D):
                m_, j = m//D, m%D
                dx = diff[n_, :, m_]
                K_hess[n:n+D, m:m+D] = (1./Gamma[i])*(E-dx[np.newaxis, :]*dx[:, np.newaxis]/Gamma[j])*K[n_, m_]
        return K_hess
    return se_hess