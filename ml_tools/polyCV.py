# Taylor expansion of kinetic energy functional

import numpy as np
import pickle
from ml_main.KernelMethod import kernelFitting, kernelRepresentedFunction

def polyKernelFunction(X, Y, degree=5):
    return (1 + X @ Y)**degree

fpath = '../data_file/quantum1D'
with open(fpath, 'rb') as dataFile:
    Data = pickle.load(dataFile)


trainDatasetSize = 5000
testDatasetSize = 3000
numOfFeature = Data.shape[1] - 1
referenceDensity = np.c_[np.ones((8000, 1))*2, np.zeros((8000, numOfFeature-1))]
dataFeature = Data[:, :numOfFeature] - referenceDensity
shiftedData = np.c_[dataFeature, Data[:, -1]]

trainDataset = shiftedData[:trainDatasetSize]
testDataset = shiftedData[trainDatasetSize:]

# training & cross validation
penalty = 10**3
trainSetIndex = np.arange(0, trainDatasetSize, 1, dtype=np.int64)
for i in range(5):
    validateIndex = np.random.choice(trainSetIndex, 1000, replace=False)
    subTrainIndex = np.setdiff1d(trainSetIndex, validateIndex)
    np.random.shuffle(subTrainIndex)
    subTrainDataset, validateDataset = trainDataset[subTrainIndex], trainDataset[validateIndex]
    subTrainFeature, subTrainTarget = subTrainDataset[:, :numOfFeature], subTrainDataset[:, -1]
    validateFeature, validateTarget = validateDataset[:, :numOfFeature], validateDataset[:, -1]

    coefficients = kernelFitting(subTrainFeature, subTrainTarget, penalty, polyKernelFunction)
    kineticEnergyFunctional = kernelRepresentedFunction(subTrainFeature, coefficients, polyKernelFunction)
    predictTarget = kineticEnergyFunctional(validateFeature)
    meanSquareError = np.linalg.norm((predictTarget - validateTarget))/1000
    print('In the %s cross-validation, the mse is %s' %(i, meanSquareError))




