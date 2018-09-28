import numpy as np
import pickle
from ml_tools.quantum import quantum1D

np.random.seed(5)

numOfKpoints = 40
numOfBasis = 31
MaxFFTComponent = 10
VqMagnitude = np.linspace(0, 50, 10)
samplingSteps = 10
DeltaMu = np.linspace(5, 50, 10)

DataStorage = []
for numOfFFTComponents in range(2, MaxFFTComponent):
    for absVq in VqMagnitude:
        for deltaMu in DeltaMu:
            for _ in range(samplingSteps):
                model = quantum1D(numOfBasis, numOfFFTComponents, absVq)
                KineticEnergyPerCell, ChemicalPotential, _, externalPotential, ElectronDensityPerCell = model(deltaMu, numOfKpoints)
                DataStorage.append(np.array([*ElectronDensityPerCell, KineticEnergyPerCell]))

Data = np.array(DataStorage)
np.random.shuffle(Data)
fname = '/home/hongbin/Documents/project/machine_learning_kinetic_energy/data_file/quantum1D'
with open(fname, 'wb') as f:
    pickle.dump(Data, f)




