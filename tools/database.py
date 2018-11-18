import numpy as np
import pickle
from MLEK.main.quantum_utils import quantum1D

np.random.seed(5)

numOfKpoints = 40
numOfBasis = 31
MaxFFTComponent = 5
LowerBound, UpperBound = -5, 5
samplingSteps = 2
DeltaMu = np.linspace(5, 50, 10)

DataStorage = []
for num_Potential_FFTcmp in range(2, MaxFFTComponent):
    for deltaMu in DeltaMu:
        for _ in range(samplingSteps):
            model = quantum1D(numOfBasis, num_Potential_FFTcmp, LowerBound, UpperBound)
            KineticEnergyPerCell, ChemicalPotential, _, externalPotential, ElectronDensityPerCell = model(deltaMu, numOfKpoints)
            DataStorage.append(np.array([*ElectronDensityPerCell, KineticEnergyPerCell]))

Data = np.array(DataStorage)
np.random.shuffle(Data)
fname = '../data_file/quantum1D'
with open(fname, 'wb') as f:
    pickle.dump(Data, f)




