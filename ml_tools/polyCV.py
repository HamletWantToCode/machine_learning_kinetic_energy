# Taylor expansion of kinetic energy functional, training and cross-validation

import numpy as np
import pickle
from ml_main.KernelMethod import kernelFitting, kernelRepresentedFunction

def polyKernelFunction(X, Y, degree=5):
    return (1 + X @ Y)**degree

fpath = '../data_file/quantum1D'
with open(fpath, 'rb') as dataFile:
    Data = pickle.load(dataFile)

totalNumOfSample = Data.shape[0]
trainDatasetSize = 5000
numOfFeature = Data.shape[1] - 1
referenceDensity = np.c_[np.ones((totalNumOfSample, 1))*2, np.zeros((totalNumOfSample, numOfFeature))]
Data -= referenceDensity
trainDataset = Data[:trainDatasetSize]

# training & cross validation p*k
penalty = 10**3
p = 5
numOfPartition = 5
numPerPartition = trainDatasetSize // numOfPartition
for i in range(p):
    dataList = trainDataset.reshape([numOfPartition, numPerPartition, -1])
    for i in range(numOfPartition):
        trainSetIndex = np.setdiff1d(range(numOfPartition), i)
        subTrainDataset, validateDataset = np.vstack(dataList[trainSetIndex]), dataList[i]
        subTrainFeature, subTrainTarget = subTrainDataset[:, :numOfFeature], subTrainDataset[:, -1]
        validateFeature, validateTarget = validateDataset[:, :numOfFeature], validateDataset[:, -1]

        coefficients = kernelFitting(subTrainFeature, subTrainTarget, penalty, polyKernelFunction)
        kineticEnergyFunctional = kernelRepresentedFunction(subTrainFeature, coefficients, polyKernelFunction)
        predictTarget = kineticEnergyFunctional(validateFeature)
        meanSquareError = np.linalg.norm((predictTarget - validateTarget))/numPerPartition
        print('In the %s cross-validation, the mse is %s' %(i, meanSquareError))
    np.random.shuffle(trainDataset)




