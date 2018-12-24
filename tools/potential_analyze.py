import pickle
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.ticker import *

with open('../data_file/potential', 'rb') as f:
    Vq = pickle.load(f)
Vx = np.fft.irfft(Vq, 100, axis=1)*100
n = Vx.shape[0]

# EU distance
distance = np.zeros(n*(n-1))
for i in range(n):
    D = np.sqrt(np.sum((Vx[i][np.newaxis, :] - np.delete(Vx, i, 0))**2, axis=1))
    distance[i*(n-1):(i+1)*(n-1)] = D
distance = np.unique(distance)

fig1 = plt.figure(1)
ax = fig1.gca()
_, bin_edges, _ = ax.hist(distance, 20, density=True)
bin_width = bin_edges[1] - bin_edges[0]

def yscale(y, position):
    return np.round(bin_width*y, 2)

funcfmt = FuncFormatter(yscale)
ax.yaxis.set_major_formatter(funcfmt)
ax.set_xlabel('distance')
ax.set_ylabel('fraction')

plt.show()
