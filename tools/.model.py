import pickle
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error as mse 

def model(fname, gamma, alpha):

    with open(fname, 'rb') as f:
        data = pickle.load(f)

    print('...processing data')
    densG = data.iloc[:, :51].values
    Ek = data.iloc[:, 51].values
    dE_dnq = data.iloc[:, 52:].values

    train_dens, test_dens, train_Ek, test_Ek, train_dEk, test_dEk = train_test_split(densG, Ek, dE_dnq, test_size=0.2, random_state=62)
    train_data = np.c_[train_dens, train_Ek.reshape(-1, 1), train_dEk]
    test_data = np.c_[test_dens, test_Ek.reshape(-1, 1), test_dEk]
    with open('../train_data', 'wb') as f1:
        pickle.dump(train_data, f1)
    with open('../test_data', 'wb') as f2:
        pickle.dump(test_data, f2)
    print('train/test data saved...')

    print('...model training')
    
    krr = KernelRidge(kernel='poly', degree=3, coef0=0, gamma=gamma, alpha=alpha)
    krr.fit(train_dens, train_Ek)
    pred_Ek = krr.predict(test_dens)
    err = mse(test_Ek, pred_Ek)
    print('model testing err: {:<=.4f}'.format(err))
    return krr