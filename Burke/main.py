# finite difference method

import numpy as np

def gaussPotential(A, B, C):
    # a three-dip Gaussian function
    def function(x):
        V = 0
        for i in range(3):
            V += -A[i]*np.exp(-(x-B[i])**2/(2*C[i]**2))
        return V
    return function

def finiteDifferenceMatrix(n, xstart, xend, potentialFunction):
    T = np.zeros((n, n), np.float64)
    V = np.zeros((n, n), np.float64)
    L = xend - xstart
    np.fill_diagonal(T, -2)
    np.fill_diagonal(T[1:, :-1], 1)
    np.fill_diagonal(T[:-1, 1:], 1)
    h = L*1.0/(n+1)
    Vx = [potentialFunction(xstart+h*(i+1)) for i in range(n)]
    np.fill_diagonal(V, Vx)
    return T*(-0.5*(n+1)**2)/L**2 + V

def electronDensity(Psi, ne, n, xstart, xend):
    # not consider spin
    density = np.zeros(n, np.float64)
    L = xend - xstart
    for i in range(ne):
        density += ((n+1)*1.0/L)*(Psi[:, i]**2)
    return np.r_[0, density, 0]

def kineticEnergy(n, xstart, xend, ne, Psi):
    Tk = np.zeros((n, n+2), np.float64)
    L = xend - xstart
    Ek = 0
    for i in range(n):
        Tk[i, i], Tk[i, i+1], Tk[i, i+2] = 1, -2, 1
    Tk *= -0.5*(n+1)**2/L**2
    for j in range(ne):
        Psi_j_withEndPoints = np.r_[0, Psi[:, j], 0]
        Ek += Psi[:, j] @ Tk @ Psi_j_withEndPoints
    return Ek

def compute(n, ne, A, B, C, xstart=0, xend=1):
    potential = gaussPotential(A, B, C)
    H = finiteDifferenceMatrix(n, xstart, xend, potential)
    _, eigenFunctions = np.linalg.eigh(H)
    density = electronDensity(eigenFunctions, ne, n, xstart, xend)
    Ek = kineticEnergy(n, xstart, xend, ne, eigenFunctions)
    return np.array([ne, Ek, *density])








