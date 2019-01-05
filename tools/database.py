# database parallel

import numpy as np
from mpi4py import MPI
import pickle
from MLEK.main.utils import simple_potential_gen
from MLEK.main.solver import solver

NSAMPLES = 100
LOW_A = 100
HIGH_A = 500
LOW_B = 0.2
HIGH_B = 0.8
LOW_C = 0.01
HIGH_C = 0.3
MU = 20
NK = 100
NBASIS = 100

comm = MPI.COMM_WORLD
SIZE = comm.Get_size()
NSAMPLE_PER_PROC = NSAMPLES // SIZE
ID = comm.Get_rank()

POTENTIAL_STORAGE = np.zeros((NSAMPLE_PER_PROC, NBASIS+1), dtype=np.complex64)
DATA_STORAGE = np.zeros((NSAMPLE_PER_PROC, NBASIS+1), dtype=np.complex64)
RANDOM_STATE = ID
param_gen = simple_potential_gen(NBASIS, LOW_A, HIGH_A, LOW_B, HIGH_B, LOW_C, HIGH_C, MU, RANDOM_STATE)
for i in range(NSAMPLE_PER_PROC):
    hamilton_mat, Vq, mu = next(param_gen)
    T, mu, density = solver(NK, NBASIS, mu, hamilton_mat)
    DATA_STORAGE[i] = np.array([T, *density], dtype=np.complex64)
    POTENTIAL_STORAGE[i] = np.array([mu, *Vq], dtype=np.complex64)

DATA = None
POTENTIAL = None
if ID == 0:
    DATA = np.zeros((NSAMPLES, NBASIS+1), dtype=np.complex64)
    POTENTIAL = np.zeros((NSAMPLES, NBASIS+1), dtype=np.complex64)

comm.Gather(DATA_STORAGE, DATA, root=0)
comm.Gather(POTENTIAL_STORAGE, POTENTIAL, root=0)

if ID == 0:
    with open('../data_file/quantum', 'wb') as f:
        pickle.dump(DATA, f)
    with open('../data_file/potential', 'wb') as f1:
        pickle.dump(POTENTIAL, f1)

