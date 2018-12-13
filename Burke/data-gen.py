# data generating

import pickle
from main import compute
import numpy as np
from mpi4py import MPI

N_dataSample = 2000
N_grid = 20
a_low, a_high = 1, 10
b_low, b_high = 0.4, 0.6
c_low, c_high = 0.03, 0.1
n_cpu = 4
N_per_script = int(N_dataSample / n_cpu)
part_data = np.zeros((N_per_script, N_grid+4))
part_potential = np.zeros((N_per_script, N_grid+3))
dataStorage = None
potentialStorage = None

comm = MPI.COMM_WORLD
ID = comm.Get_rank()
np.random.seed(ID)

for i in range(N_per_script):
    A = np.random.uniform(a_low, a_high, 3)
    B = np.random.uniform(b_low, b_high, 3)
    C = np.random.uniform(c_low, c_high, 3)
    N_electron = np.random.randint(1, 5)
    data, potential = compute(N_grid, N_electron, A, B, C)
    part_data[i] = data
    part_potential[i] = potential

if ID == 0:
    dataStorage = np.zeros((N_dataSample, N_grid+4), np.float64)
    potentialStorage = np.zeros((N_dataSample, N_grid+3), np.float64)

comm.Gather(part_data, dataStorage)
comm.Gather(part_potential, potentialStorage)

if ID == 0:
    with open('quantumX1D', 'wb') as f:
        pickle.dump(dataStorage, f)
    with open('potentialX1D', 'wb') as f1:
        pickle.dump(potentialStorage, f1)
