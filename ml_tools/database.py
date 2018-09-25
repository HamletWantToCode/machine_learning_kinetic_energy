import numpy as np 
import pandas as pd 
import pickle
from quantum import quantum

nk = 81
nG = 51   # odd number !!
N = 500

data_nqTdT = []

np.random.seed(336)     # repeatable

V_s = np.arange(20, 40.1, 0.1)       # range of the strength of potential
V_FFT_components = [1, 2, 3, 4, 5, 6]
dMu = np.arange(40, 60.1, 0.1)

for i in range(N):
    V0 = np.random.choice(V_s)
    n_cmp = np.random.choice(V_FFT_components)
    dmu = np.random.choice(dMu)
    Ek, mu, _, Vq, densG = quantum(n_cmp, V0, dmu, nk, nG)
    Mu = np.zeros_like(Vq)
    Mu[0] = mu
    dTdn = Mu - Vq
    data_nqTdT.append([*densG, Ek, *dTdn])

df_nqTdT = pd.DataFrame(data_nqTdT)

fname = '../densG_T_dT'
with open(fname, 'wb') as f:
    pickle.dump(df_nqTdT, f)



    
