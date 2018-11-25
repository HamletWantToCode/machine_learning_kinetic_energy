# database generate

import numpy as np 
from MLEK.main.base import Compute
from MLEK.main.utils import potential_gen
from MLEK.main.solver import solver

LOW_NUM_Q = 1
HIGH_NUM_Q = 10
LOW_V0 = 0          # absolute value
HIGH_V0 = 50
LOW_MU = 5
HIGH_MU = 20
RANDOM_STATE = 5

param_gen = potential_gen(LOW_NUM_Q, HIGH_NUM_Q, LOW_V0, HIGH_V0, LOW_MU, HIGH_MU, RANDOM_STATE)
data = Compute(2, 100, '../data_file/quantum_data', '../data_file/potential_data')
data.add_solver(solver)

nk = 40
nbasis = 30
data.run(nk, nbasis, param_gen)
