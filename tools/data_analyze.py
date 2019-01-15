import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import *
import matplotlib.gridspec as gridspec

with open('../data_file/quantum', 'rb') as f:
    data = pickle.load(f)
dens_q, Ek = data[:, 1:], data[:, 0].real
n = dens_q.shape[0]

# EU distance
distance = np.zeros(n*(n-1))
for i in range(n):
    Z = dens_q[i][np.newaxis, :] - np.delete(dens_q, i, 0)
    D = np.sqrt(np.sum(Z.conj()*Z, axis=1))
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
mean_x = np.mean(dens_q, axis=0)
Cov = ((dens_q - mean_x).conj()).T @ (dens_q - mean_x) / n
U, S, Uh = np.linalg.svd(Cov)
dens_qt = (dens_q - mean_x) @ U
real_dens_qt = dens_qt.imag

## display the range of first 5 dimension
fig2 = plt.figure(2)
gds = gridspec.GridSpec(5, 1)
for i in range(5):
    ax = fig2.add_subplot(gds[i])
    ax.hist(real_dens_qt[:, i], 20, density=True, label='principal #%d' %(i))
    ax.yaxis.set_major_formatter(funcfmt)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.legend(loc=1)
fig2.text(0.5, 0.04, 'range', ha='center')
fig2.text(0.04, 0.5, 'fraction', va='center', rotation='vertical')

## 2D display of data distribution
fig4 = plt.figure(4)
ax = fig4.gca()
ax.scatter(real_dens_qt[:, 0], real_dens_qt[:, 1], c='b')
ax.set_xlabel('principal #1')
ax.set_ylabel('principal #2')

## correlation
corr_coef = np.zeros(20)
for i in range(20):
    corr = np.corrcoef(real_dens_qt[:, i], Ek)[1, 0]
    corr_coef[i] = corr

fig3 = plt.figure(3)
ax_corr = fig3.gca()
ax_corr.plot(np.arange(0, 20, 1), corr_coef, 'bo-')
ax_corr.xaxis.set_major_locator(MultipleLocator(1.0))
ax_corr.set_xlabel('dimension')
ax_corr.set_ylabel('correlation coefficient')

plt.show()

