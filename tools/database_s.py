# database serial

import numpy as np
import pickle
from MLEK.main.utils import potential_gen
from MLEK.main.solver import solver

NSAMPLES = 500
LOW_NUM_Q = 1
HIGH_NUM_Q = 30
LOW_V0 = 0          # absolute value
HIGH_V0 = 50
LOW_dMU = 5
HIGH_dMU = 50
NK = 100
NBASIS = 100

POTENTIAL_STORAGE = np.zeros((NSAMPLES, NBASIS+1), dtype=np.complex64)
DATA_STORAGE = np.zeros((NSAMPLES, NBASIS+1), dtype=np.complex64)
RANDOM_STATE = 5
param_gen = potential_gen(NBASIS, LOW_NUM_Q, HIGH_NUM_Q, LOW_V0, HIGH_V0, LOW_dMU, HIGH_dMU, RANDOM_STATE)
for i in range(NSAMPLES):
    dmu, hamilton_mat, Vq = next(param_gen)
    T, mu, density = solver(NK, NBASIS, dmu, hamilton_mat)
    DATA_STORAGE[i] = np.array([T, *density], dtype=np.complex64)
    POTENTIAL_STORAGE[i] = np.array([mu, *Vq], dtype=np.complex64)

with open('../data_file/quantum', 'wb') as f:
    pickle.dump(DATA_STORAGE, f)
with open('../data_file/potential', 'wb') as f1:
    pickle.dump(POTENTIAL_STORAGE, f1)
