import numpy as np
import pickle
import matplotlib.pyplot as plt 

with open('quantumX1D', 'rb') as f:
    dens = pickle.load(f)

with open('potentialX1D', 'rb') as f1:
    potential = pickle.load(f1)

dens_n1 = dens[dens[:, 0]==1, 2:]
n = dens_n1.shape[0]
Ek_n1 = dens[dens[:, 0]==1, 1]
potential_n1 = potential[potential[:, 0]==1, 1:]

mean_X = np.mean(dens_n1, axis=0)
Cov = (dens_n1 - mean_X).T @ (dens_n1 - mean_X) / n
U, S, _ = np.linalg.svd(Cov)

dens_1D = (dens_n1 - mean_X) @ U[:, 20]
y = Ek_n1
dy = -potential_n1 @ U[:, 20]

plt.plot(dens_1D, dy, 'bo')
plt.show()
