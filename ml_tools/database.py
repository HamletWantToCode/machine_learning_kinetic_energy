import numpy as np
import pandas as pd
import pickle
from ml_tools.quantum import quantum1D

np.random.seed(336)

nk = 80
nG = 51
num_VStrength = 20
num_Vcomponents = 10
num_deltaMu = 20

DataStorage = []

PotentialStrength = np.linspace(0, 100, num_VStrength)
PotentialComponents = np.arange(1, num_Vcomponents, 1)
DeltaMu = np.linspace(5, 50, num_deltaMu)
X, Y, Z = np.meshgrid(PotentialStrength, PotentialComponents, DeltaMu, indexing='ij')
parameters = list(zip(X.flatten(), Y.flatten(), Z.flatten()))

for Vs, n_cmp, dmu in parameters:
    Ek, mu, _, Vq, densG = quantum1D(n_cmp, Vs, dmu, nk, nG)
    DataStorage.append([*densG, Ek])

Data = pd.DataFrame(DataStorage)

fname = '/home/hongbin/Documents/project/machine_learning_kinetic_energy/data_file/quantum1D'
with open(fname, 'wb') as f:
    pickle.dump(Data, f)




