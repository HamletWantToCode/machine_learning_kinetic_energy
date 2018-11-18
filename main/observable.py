import numpy as np 

def density_operator_bloch_1D(ink: int, nG: int) -> np.matrix:
    n_kG = np.r_[np.ones(ink), np.zeros(nG-ink)]
    return np.diag(n_kG)

# def density_operator_bloch_2D(ink: int, nG: int) -> np.matrix:
#     n_kG = np.r_[np.ones(ink), np.zeros(nG**2-ink)]
#     return np.diag(n_kG)

def kinetic_operator_G_1D(k: float, nG: int) -> np.matrix:
    T_nk = []
    for i in range(nG):
        T_nk.append(0.5*(k+2*np.pi*(i-nG//2))**2)
    return np.diag(T_nk)

# def kinetic_operator_G_2D(kx, ky, nG):
#     T_nk = np.zeros((nG**2, nG**2))
#     for i in range(nG**2):
#         m_, n_ = i//nG, i%nG
#         T_nk[i, i] = 0.5*((kx+2*np.pi*(m_-nG//2))**2 + (ky+2*np.pi*(n_-nG//2))**2)
#     return T_nk

def density1D(nk, nG, mu, en_band, uq):
    rd_fstBz = np.linspace(0, np.pi, nk)
    n_G = np.zeros((nG, nG), np.complex64)
    for i, kx in enumerate(rd_fstBz):
        ens = en_band[i]
        occ_n = len(ens[ens<=mu])
        nk_bloch = density_operator_bloch_1D(occ_n, nG)
        uq_k = uq[i]
        Huq_k = np.conjugate(np.transpose(uq_k))
        n_kG = uq_k @ nk_bloch @ Huq_k
        n_G += n_kG
    rho_G = np.zeros(nG)
    for i in range(nG):
        rho_G[0] += n_G[i, i]
        for j in range(i+1, nG):
            if (j-i)<=(nG-1)//2:
                rho_G[j-i] += n_G[j, i]
                rho_G[i-j] += n_G[i, j]
    return rho_G / nk

def kinetic_en1D(nk, nG, mu, en_band, uq):
    rd_fstBz = np.linspace(0, np.pi, nk)
    kinetic_en = 0
    for i, kx in enumerate(rd_fstBz):
        Tn_kG = kinetic_operator_G_1D(kx, nG)
        ens = en_band[i]
        occ_n = len(ens[ens<=mu])
        for ni in range(occ_n):
            un_q = uq[i, :, ni]
            Hun_q = np.conjugate(un_q.T)
            T_nk = Hun_q @ Tn_kG @ un_q
            kinetic_en += T_nk
    return kinetic_en.real / nk

# def kinetic_en2D(nk, nG, mu, en_band, uq):
#     rd_fstBz = np.linspace(0, np.pi, nk)
#     kinetic_en = 0
#     for i, kx in enumerate(rd_fstBz):
#         for j, ky in enumerate(rd_fstBz):
#             Tn_kG = kinetic_operator_G_2D(kx, ky, nG)
#             ens = en_band[i, j]
#             occ_n = len(ens[ens<=mu])
#             for ni in range(occ_n):
#                 un_q = uq[i, j, :, ni]
#                 Hun_q = np.conjugate(un_q.T)
#                 T_nk = Hun_q @ Tn_kG @ un_q
#                 kinetic_en += T_nk
#     return kinetic_en.real / nk**2

# def density2D(nk, nG, mu, en_band, uq):
#     rd_fstBz = np.linspace(0, np.pi, nk)
#     n_G = np.zeros((nG**2, nG**2), np.complex64)
#     for i, kx in enumerate(rd_fstBz):
#         for j, ky in enumerate(rd_fstBz):
#             ens = en_band[i, j]
#             occ_n = len(ens[ens<=mu])
#             uk_G = uq[i, j]
#             Huk_G = np.conjugate(np.transpose(uk_G))
#             nk_bloch = density_operator_bloch_2D(occ_n, nG)
#             nk_G = uk_G @ nk_bloch @ Huk_G
#             n_G += nk_G
#     n_G4 = n_G.reshape((nG, nG, nG, nG))
#     densG = np.zeros((nG, nG))
#     for i in range(nG**2):
#         m_, n_ = i//nG, i%nG
#         densG[0, 0] += n_G4[m_, n_, m_, n_]
#         for j in range(i+1, nG**2):
#             m1_, n1_ = j//nG, j%nG
#             if (m1_-m_)<=(nG-1)//2 and abs(n1_-n_)<=(nG-1)//2:
#                 if n1_ == n_:
#                     densG[m1_-m_, 0] += n_G4[m1_, n_, m_, n_]
#                     densG[m_-m1_, 0] += n_G4[m_, n_, m1_, n_]
#                 if n1_>n_:
#                     densG[m1_-m_, n1_-n_] += n_G4[m1_, n1_, m_, n_]
#                     densG[m_-m1_, n_-n1_] += n_G4[m_, n_, m1_, n1_]
#     return densG / nk**2




