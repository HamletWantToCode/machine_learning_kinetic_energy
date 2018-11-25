# potential generator

import numpy as np 

def potential_gen(low_nq, high_nq, low_V0, high_V0, low_mu, high_mu, random_state, maxiters=10000):
    np.random.seed(random_state)
    i = 1
    while True:
        if i>maxiters:
            print('run out of the maximum iterations !')
            break
        nq = np.random.randint(low_nq, high_nq)
        Vq = np.zeros(nq, dtype=np.complex64)
        V0 = np.random.uniform(low_V0, high_V0)
        for i in range(1, nq):
            theta = np.random.uniform(0, 2*np.pi)
            Vq[i] = -V0*(np.cos(theta) + 1j*np.sin(theta))
        mu = np.random.uniform(low_mu, high_mu)
        yield (mu, Vq)
        i += 1