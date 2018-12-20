# curse of dimensionality

import pickle
import numpy as np 
from scipy.special import gamma
import matplotlib.pyplot as plt 
from matplotlib import ticker

with open('/Users/hongbinren/Downloads/mnt/project/ML_periodic/quantum', 'rb') as f:
    data = pickle.load(f)

dens_q = data[:, 1:]
dens_X = np.fft.irfft(dens_q, 100, axis=1)*100
mean_dens = np.mean(dens_X, axis=0)
n_dens = data.shape[0]

Cov = (dens_X - mean_dens).T @ (dens_X - mean_dens) / n_dens
U, _, _ = np.linalg.svd(Cov)

def knn(x, X, n):
    d = x.shape[1]
    D = np.sqrt(np.sum((X - x)**2, axis=1))
    D.sort()
    r = D[n+1]
    return (np.pi**(d*0.5))*r**d/gamma(0.5*d+1)

V = []
for i in range(1, 21):
    dens_Xt = (dens_X - mean_dens) @ U[:, :i]
    dens_center = np.mean(dens_Xt, axis=0, keepdims=True)
    V.append(knn(dens_center, dens_Xt, 50))

ax = plt.gca()
ax.plot(np.arange(1, 21, 1), V, 'bo-')
ax.set_xlabel('dimension')
ax.set_ylabel('Volumn')
# ax.semilogy()
majors = np.arange(1, 25, 4)
ax.xaxis.set_major_locator(ticker.FixedLocator(majors))
plt.show()
# plt.savefig('data_file/curse_dim.png')
