import numpy as np

# potential generator
def potential_gen(nbasis, nq, max_q, low_V0, high_V0, ne, random_state):
    np.random.seed(random_state)
    NG = np.arange(1, max_q, 1)
    while True:
        # if i>maxiters:
        #     print('run out of the maximum iterations !')
        #     break
        # nq = np.random.randint(low_nq, high_nq)
        q_index = np.random.choice(NG, size=nq)
        Vq = np.zeros(nbasis, dtype=np.complex64)
        hamilton_mat = np.zeros((nbasis, nbasis), dtype=np.complex64)
        V0 = np.random.uniform(low_V0, high_V0)
        for i in q_index:
            theta = np.random.uniform(0, 2*np.pi)
            Vq_conj = -V0*(np.cos(theta) - 1j*np.sin(theta))
            Vq[i] = Vq_conj.conjugate()
            np.fill_diagonal(hamilton_mat[i:, :-i], Vq_conj)
        # dmu = np.random.uniform(low_dmu, high_dmu)
        yield (hamilton_mat, Vq)
        # i += 1

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
