# various spatial distribution & prediction error

import pickle
import numpy as np  
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from matplotlib.ticker import *

with open('quantumX1D', 'rb') as f:
    data = pickle.load(f)
with open('potentialX1D', 'rb') as f1:
    potential = pickle.load(f1)
ne = 1
dens_X, Ek, dEk = data[data[:, 0]==ne, 2:], data[data[:, 0]==ne, 1], -potential[data[:, 0]==ne, 1:]
mean_X = np.mean(dens_X, axis=0)
n = dens_X.shape[0]

# spatial scale in each dimension
Cov = (dens_X - mean_X).T @ (dens_X - mean_X) / n
U, S, _ = np.linalg.svd(Cov)
dens_Xt = (dens_X - mean_X) @ U

fig1, ax1 = plt.subplots(1, 1, figsize=(5, 5))
Corr = []
for i in range(20):
    corr = np.corrcoef(dens_Xt[:, i], Ek)[1, 0]
    Corr.append(corr)
ax1.plot(np.arange(0, 20, 1), Corr, 'bo-')
ax1.set_xlabel('dimension')
ax1.xaxis.set_major_locator(FixedLocator(np.arange(0, 22, 2)))
ax1.set_ylabel('corrlation coefficient')

# distribution of data in each dimension
def topercent(y, position):
    pos = np.round((bin_width*y), 2)
    return pos
funcfmt = FuncFormatter(topercent)

fig2 = plt.figure(2)
gds = gridspec.GridSpec(5, 1)
for i in range(5):
    ax = fig2.add_subplot(gds[i])
    ax.hist(dens_Xt[:, i], 20, density=True, label='principal #%d' %(i))
    ax.yaxis.set_major_formatter(funcfmt)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.legend(loc=1)
fig2.text(0.5, 0.04, 'range', ha='center')
fig2.text(0.04, 0.5, 'fraction', va='center', rotation='vertical')

plt.show()
