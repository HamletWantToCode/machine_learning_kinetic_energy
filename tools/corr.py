import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import *

with open('quantum', 'rb') as f:
    data = pickle.load(f)
dens_X, Ek = np.fft.irfft(data[:, 1:], 100, axis=1)*100, data[:, 0].real
mean_X = np.mean(dens_X, axis=0)
n = dens_X.shape[0]

# spatial scale in each dimension
Cov = (dens_X - mean_X).T @ (dens_X - mean_X) / n
U, S, _ = np.linalg.svd(Cov)
dens_Xt = (dens_X - mean_X) @ U

fig = plt.figure()
gds = GridSpec(5, 2, figure=fig)
axes = [fig.add_subplot(gds[i, j]) for i in range(5) for j in range(2)]
for i in range(1, 11):
    ax = fig.add_subplot(axes[i-1])
    ax.hist(dens_Xt[:, i-1], 20)
    ax.set_xlim([-10, 10])
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
axes[8].xaxis.set_major_locator(FixedLocator([-10, 10]))
axes[8].xaxis.set_major_formatter(FixedFormatter([-10, 10]))
axes[9].xaxis.set_major_locator(FixedLocator([-10, 10]))
axes[9].xaxis.set_major_formatter(FixedFormatter([-10, 10]))

fig.text(0.5, 0.04, 'range')
fig.text(0.04, 0.5, 'fraction', va='center', rotation='vertical')
plt.show()
