import numpy as np 
import pandas as pd 
import pickle
from ml_tools.quantum import quantum1D

np.random.seed(336)

nk = 80
nG = 31
num_VStrength = 10
num_Vcomponents = 5
num_deltaMu = 3

DataStorage = []

PotentialStrength = np.linspace(10, 100, num_VStrength)
PotentialComponents = np.arange(1, num_Vcomponents, 1)
DeltaMu = np.linspace(5, 50, num_deltaMu)
X, Y, Z = np.meshgrid(PotentialStrength, PotentialComponents, DeltaMu, indexing='ij')
parameters = list(zip(X.flatten(), Y.flatten(), Z.flatten()))

for Vs, n_cmp, dmu in parameters:
    model = quantum1D(Vs, nG, n_cmp)
    Ek, mu, _, Vq, densG = model(dmu, nk)
    DataStorage.append([*densG, Ek])

Data = pd.DataFrame(DataStorage)

fname = '/Users/hongbinren/Documents/program/machine_learning_kinetic_energy/data_file/quantum1D'
with open(fname, 'wb') as f:
    pickle.dump(Data, f)



    
