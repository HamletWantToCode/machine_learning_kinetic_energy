# API for ed

import numpy as np
from .ed import solver1D
from .observable import kinetic_en1D, density1D
import multiprocessing as mp

def externalPotential1D(numOfBasis, numOfFFTComponents, lowerbound, upperbound):
    """
    We set Vq[0] = 0, since it only contribute to the diagonal term 
    of the Hamiltonian matrix therefore won't leads to change in electron
    density. The Fourier components of periodic potential belonges to the
    l^2 space, we are sampling the potential evenly from that space.
    : numOfBasis: odd number
    : numOffFFTComponents: < numOfBasis
    """
    len_Vq = (numOfBasis-1)//2
    Vq = np.zeros(int(len_Vq), np.complex64)
    for i in range(1, numOfFFTComponents):
        real, img =np.random.uniform(lowerbound, upperbound, 2)
        Vq[i] = real + 1j*img
    return Vq
    
# def Vextq_2D(nx, ny, nG, V0):
#     assert nG>2*ny and nG>2*nx
#     Vq = np.zeros((nG, nG//2+1), np.complex64)
#     Vq[0, 0] = np.random.rand()
#     n_ii, n_jj = np.meshgrid(range(nx), range(ny), indexing='ij')
#     n_ij = zip(n_ii.flatten(), n_jj.flatten())
#     useless = n_ij.__next__()
#     for i, j in n_ij:
#         Vq[i, j] = Vq[0, 0]*np.exp(1j*np.random.uniform(-np.pi, np.pi))
#         if i != 0:
#             Vq[-i, j] = np.conjugate(Vq[i, j])
#     Vx = np.fft.irfft2(Vq, (nG, nG))*nG**2
#     Vx_std = potential_scaler(Vx)
#     return np.fft.rfft2(Vx_std*V0)/nG**2

def quantum1D(numOfBasis, numOfPotentialFFTcmp, PotentialcmpLowerbound, PotentialcmpUpperbound):
    Vq = externalPotential1D(numOfBasis, numOfPotentialFFTcmp, PotentialcmpLowerbound, PotentialcmpUpperbound)

    def compute(shiftInChemicalPotential, numOfkPoints):
        rd_fstBz = np.linspace(0, np.pi, numOfkPoints)
        n_cpu = mp.cpu_count()

        bandDataQueue, wavefuncDataQueue = mp.Queue(), mp.Queue()
        chunk_size = numOfkPoints//n_cpu
        Procs = []
        for p_ix in range(n_cpu):
            part_kpoints = rd_fstBz[p_ix*chunk_size : (p_ix+1)*chunk_size]
            p = mp.Process(target=solver1D, args=(part_kpoints, numOfBasis, Vq, p_ix, bandDataQueue, wavefuncDataQueue))
            p.start()
            Procs.append(p)

        bandResults, wavefuncResults = [], []
        for p in Procs:
            bandResults.append(bandDataQueue.get())
            wavefuncResults.append(wavefuncDataQueue.get())
        bandResults.sort()
        wavefuncResults.sort()
        bandDataStorage = np.vstack([item[1] for item in bandResults])
        wavefuncDataStorage = np.vstack([item[1] for item in wavefuncResults])

        mu = shiftInChemicalPotential + bandDataStorage[:, 0].min()
        Ek = kinetic_en1D(numOfkPoints, numOfBasis, mu, bandDataStorage, wavefuncDataStorage)
        densG = density1D(numOfkPoints, numOfBasis, mu, bandDataStorage, wavefuncDataStorage)
        results = [Ek, mu, bandDataStorage, Vq, densG]
        return results
    return compute

# def quantum2D(nx, ny, V0, dmu, nk, nG, comm):
#     """
#     nA should larger than 1 !
#     """
#     np.random.seed(35)

#     Vq = Vextq_2D(nx, ny, nG, V0)
#     Kx = Ky = np.linspace(0, np.pi, nk)
#     fstBz_kx, fstBz_ky = np.meshgrid(Kx, Ky, indexing='ij')        # pay attention to meshgrid indexing, here I use matrix indexing 'ij'
#     rd_fstBz = list(zip(fstBz_kx.flatten(), fstBz_ky.flatten()))

#     rank, size = comm.Get_rank(), comm.Get_size()
#     m = len(rd_fstBz) // size
#     en_band_, uq_ = np.zeros((size, m, nG**2)), np.zeros((size, m, nG**2, nG**2), np.complex64)
#     # dtype CANNOT BE SET to complex !!!

#     kpoints = rd_fstBz[rank*m:(rank+1)*m]
#     part_band, part_uq = ed.solver2D(kpoints, nG, Vq)
#     comm.Gather(part_band, en_band_)
#     comm.Gather(part_uq, uq_)

#     # --------------- serial part ---------------------- #
#     if rank == 0:
#         en_band, uq = en_band_.reshape((nk, nk, nG**2)), uq_.reshape((nk, nk, nG**2, nG**2))
#         # -------- compute the lowest band energy ----- #
#         mu = dmu + en_band[:, :, 0].min()
#         # en_band0, _ = ed.solver2D([(0.0, 0.0)], nG, Vq)
#         # E0 = en_band0[0, 0]
#         # mu = E0 + dmu
#         # ------- Ek, density ----------- #
#         Ek = quantum_util.kinetic_en2D(nk, nG, mu, en_band, uq)
#         densG = quantum_util.density2D(nk, nG, mu, en_band, uq)
#         results = [Ek, mu, en_band, Vq, densG]
#         with open('../data_file/quantum2D', 'wb') as f:
#             pickle.dump(results, f)
#     # --------------------------------------------------- #
#     return None
