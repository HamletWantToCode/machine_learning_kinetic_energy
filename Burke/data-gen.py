# data generating

import pickle
from main import compute
import numpy as np
from mpi4py import MPI

N_dataSample = 2000
N_grid = 500
a_low, a_high = 1, 10
b_low, b_high = 0.4, 0.6
c_low, c_high = 0.03, 0.1
n_cpu = 4
N_per_script = int(N_dataSample / n_cpu)
part_data = np.zeros((N_per_script, N_grid+3))

comm = MPI.COMM_WORLD
ID = comm.Get_rank()
np.random.seed(ID)

for i in range(N_per_script):
    A = np.random.uniform(a_low, a_high, 3)
    B = np.random.uniform(b_low, b_high, 3)
    C = np.random.uniform(c_low, c_high, 3)
    N_electron = np.random.randint(1, 5)
    data = compute(N_grid, N_electron, A, B, C)
    part_data[i] = data

if ID == 0:
    dataStorage = np.zeros((N_dataSample, N_grid+3), np.float64)

comm.Gather(part_data, dataStorage)

if ID == 0:
    with open('quantumX1D', 'wb') as f:
        pickle.dump(dataStorage, f)
