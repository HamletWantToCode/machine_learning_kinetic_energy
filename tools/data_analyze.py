import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import *
import matplotlib.gridspec as gridspec

with open('../data_file/quantum', 'rb') as f:
    data = pickle.load(f)
dens_q, Ek = data[:, 1:], data[:, 0].real
dens_x = np.fft.irfft(dens_q, 500, axis=1)*500
n = dens_x.shape[0]

# EU distance
distance = np.zeros(n*(n-1))
for i in range(n):
    D = np.sqrt(np.sum((dens_x[i][np.newaxis, :] - np.delete(dens_x, i, 0))**2, axis=1))
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

# PCA
mean_x = np.mean(dens_x, axis=0)
Cov = (dens_x - mean_x).T @ (dens_x - mean_x) / n
U, S, _ = np.linalg.svd(Cov)
dens_Xt = (dens_x - mean_x) @ U

## display the range of first 5 dimension
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

## 2D display of data distribution
fig4 = plt.figure(4)
ax = fig4.gca()
ax.scatter(dens_Xt[:, 0], dens_Xt[:, 1], c='b')
ax.set_xlabel('principal #1')
ax.set_ylabel('principal #2')

## correlation
corr_coef = np.zeros(20)
for i in range(20):
    corr = np.corrcoef(dens_Xt[:, i], Ek)[1, 0]
    corr_coef[i] = corr

fig3 = plt.figure(3)
ax_corr = fig3.gca()
ax_corr.plot(np.arange(0, 20, 1), corr_coef, 'bo-')
ax_corr.xaxis.set_major_locator(MultipleLocator(1.0))
ax_corr.set_xlabel('dimension')
ax_corr.set_ylabel('correlation coefficient')

plt.show()

