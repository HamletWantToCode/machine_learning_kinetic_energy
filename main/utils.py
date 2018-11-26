import numpy as np 

# potential generator
def potential_gen(nbasis, low_nq, high_nq, low_V0, high_V0, low_dmu, high_dmu, random_state, maxiters=10000):
    np.random.seed(random_state)
    i = 1
    while True:
        if i>maxiters:
            print('run out of the maximum iterations !')
            break
        nq = np.random.randint(low_nq, high_nq)
        Vq = np.zeros(nbasis, dtype=np.complex64)
        hamilton_mat = np.zeros((nbasis, nbasis), dtype=np.complex64)
        V0 = np.random.uniform(low_V0, high_V0)
        for i in range(1, nq):
            theta = np.random.uniform(0, 2*np.pi)
            Vq_conj = -V0*(np.cos(theta) - 1j*np.sin(theta))
            Vq[i] = Vq_conj.conjugate()
            np.fill_diagonal(hamilton_mat[i:, :-i], Vq_conj)
        dmu = np.random.uniform(low_dmu, high_dmu)
        yield (dmu, hamilton_mat, Vq)
        i += 1

# math external * under debugging !
def irfft(Aq, n_out):
    X = np.linspace(0, 1, n_out)
    nq = len(Aq)
    ifft_mat = np.zeros((n_out, nq), dtype=np.complex64)
    for i in range(n_out):
        for k in range(nq):
            ifft_mat[i, k] = np.cos(2*np.pi*k*X[i])
    ifft_mat[:, 1:] *= 2
    Ax = ifft_mat @ Aq
    return Ax.real