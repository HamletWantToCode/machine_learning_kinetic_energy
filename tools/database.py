# database parallel

import numpy as np
from mpi4py import MPI
import pickle
from MLEK.main.utils import potential_gen
from MLEK.main.solver import solver

NSAMPLES = 1000
MAX_Q = 10
# LOW_V0 = 1          # absolute value
# HIGH_V0 = 30
V0 = 10
NE = 1
# LOW_dMU = 5
# HIGH_dMU = 50
NK = 100
NBASIS = 20

comm = MPI.COMM_WORLD
SIZE = comm.Get_size()
NSAMPLE_PER_PROC = NSAMPLES // SIZE
ID = comm.Get_rank()

POTENTIAL_STORAGE = np.zeros((NSAMPLE_PER_PROC, NBASIS), dtype=np.complex64)
DATA_STORAGE = np.zeros((NSAMPLE_PER_PROC, NBASIS+1), dtype=np.complex64)
RANDOM_STATE = ID
param_gen = potential_gen(NBASIS, MAX_Q, V0, NE, RANDOM_STATE)
for i in range(NSAMPLE_PER_PROC):
    hamilton_mat, Vq, ne = next(param_gen)
    T, mu, density = solver(NK, NBASIS, ne, hamilton_mat)
    DATA_STORAGE[i] = np.array([T, *density], dtype=np.complex64)
    Vq[0] += -mu
    POTENTIAL_STORAGE[i] = Vq

DATA = None
POTENTIAL = None
if ID == 0:
    DATA = np.zeros((NSAMPLES, NBASIS+1), dtype=np.complex64)
    POTENTIAL = np.zeros((NSAMPLES, NBASIS), dtype=np.complex64)

comm.Gather(DATA_STORAGE, DATA, root=0)
comm.Gather(POTENTIAL_STORAGE, POTENTIAL, root=0)

if ID == 0:
    with open('../data_file/quantum', 'wb') as f:
        pickle.dump(DATA, f)
    with open('../data_file/potential', 'wb') as f1:
        pickle.dump(POTENTIAL, f1)

