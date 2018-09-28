# Taylor expansion of kinetic energy functional

import numpy as np 
import pickle
from ml_main.KernelMethod import kernelFitting, kernelRepresentedFunction

def polyKernelFunction(X, Y, degree=3):
    return (1 + X @ Y)**degree

fpath = '/Users/hongbinren/Documents/program/machine_learning_kinetic_energy/data_file'
with open(fpath, 'rb') as dataFile:
    Data = pickle.load(dataFile)

trainDatasetSize = 5000
testDatasetSize = 3000
numOfFeature = Data.shape[1] - 1

trainDataset = Data[:trainDatasetSize]
validateDataset = Data[trainDatasetSize : validateDatasetSize]
# testDataset = Data[(trainDatasetSize + validateDatasetSize) : testDatasetSize]

# training & cross validation
penalty = 10**3
referenceDensity = np.c_[(np.ones(trainDatasetSize)*2).reshape(-1, 1), np.zeros((trainDatasetSize, numOfFeature-1))]

trainSetIndex = np.arange(0, trainDatasetSize, 1, dtype=np.int64)
for i in range(5):
    validateIndex = np.random.choice(trainSetIndex, 1000, replace=False)
    subTrainIndex = np.setdiff1d(trainSetIndex, validateIndex)
    np.random.shuffle(subTrainIndex)
    subTrainDataSet, validateDataset = trainDataset[subTrainIndex], trainDataset[validateIndex]
    subTrainFeature, subTrainTarget = subTrainDataset[:, :numOfFeature] - referenceDensity, subTrainDataset[:, -1]
    validateFeature, validateTarget = validateDataset[:, :numOfFeature] - referenceDensity, validateDataset[:, -1]

    coefficients = kernelFitting(subTrainFeature, subTrainTarget, penalty, polyKernelFunction)
    kineticEnergyFunctional = kernelRepresentedFunction(subTrainFeature, coefficients, polyKernelFunction)
    predictTarget = kineticEnergyFunctional(validateFeature)
    meanSquareError = np.linalg.norm((predictTarget - validateTarget))
    print('In the %s cross-validation, the mse is %s' %(i, meanSquareError))




