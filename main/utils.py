import numpy as np

# potential generator
def potential_gen(nbasis, max_q, V0, mu, random_state):
    np.random.seed(random_state)
    assert max_q > 2
    NG = np.arange(2, max_q, 1, 'int')
    while True:
        nq = np.random.randint(0, max_q-2)      # nq is number of non-zero k components other than 0 and 1 component
        if nq == 0:
            q_index = np.array([1])
        else:
            q_index = np.append(np.random.choice(NG, size=nq), 1)
        Vq = np.zeros(nbasis, dtype=np.complex64)
        hamilton_mat = np.zeros((nbasis, nbasis), dtype=np.complex64)
        for i in q_index:
            theta = np.random.uniform(0, 2*np.pi)
            r0 = np.random.rand()
            Vq_conj = -V0*r0*(np.cos(theta) - 1j*np.sin(theta))
            Vq[i] = Vq_conj.conjugate()
            np.fill_diagonal(hamilton_mat[i:, :-i], Vq_conj)
        yield (hamilton_mat, Vq, mu)

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
