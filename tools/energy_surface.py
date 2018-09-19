# Ploting energy surface around the real ground state density

import numpy as np 
from QML.main import kinetic_energy_functional as Ekf

def hamilton(densG, Vq, mu, *args):
    assert len(Vq) == len(densG)
    gamma, alpha, Xi = args
    Ek = Ekf.kinetic_dens(densG.reshape(1, -1), alpha, Xi, gamma)
    dens_neq = np.r_[densG[0], densG[1:][::-1]]
    Mu = np.zeros_like(densG)
    Mu[0] = mu
    return Ek + ((Vq - Mu) @ dens_neq).real

def energy_surface(i, j, mu, gamma, densG, Vq, alpha, Xi):
    E = []
    dx = np.linspace(-30, 20, 100)
    dy = np.linspace(-30, 20, 100)
    dx_, dy_ = np.meshgrid(dx, dy)
    dr = zip(dx_.reshape(-1), dy_.reshape(-1))
    for x, y in dr:
        e = np.zeros_like(densG)
        e[i], e[j] = x, y
        densG_nn = densG + e
        E.append(hamilton(densG_nn, Vq, mu, gamma, alpha, Xi))
    E = np.array(E).reshape(dx_.shape)
    return E, dx_, dy_
    



