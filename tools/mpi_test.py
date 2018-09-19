import numpy as np 
from quantum import quantum2D
from mpi4py import MPI
import time

comm = MPI.COMM_WORLD

nk = 40
nG = 11
dmu = 50
V0 = 0
nx = 2
ny = 5
start_time = time.time()
quantum2D(nx, ny, V0, dmu, nk, nG, comm)
end_time = time.time()
print('%.4fs' %(end_time - start_time))

