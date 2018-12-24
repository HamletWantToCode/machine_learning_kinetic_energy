import numpy as np 
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import *
from mpl_toolkits.axes_grid1 import ImageGrid

with open('../data_file/quantum', 'rb') as f:
    data = pickle.load(f)
dens_q, Ek = data[:, 1:], data[:, 0].real
dens_x = np.fft.irfft(dens_q, 100, axis=1)*100
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

fig2 = plt.figure(2)
main_rect = [0.1, 0.1, 0.6, 0.6]
up_rect = [0.1, 0.71, 0.6, 0.2]
right_rect = [0.71, 0.1, 0.2, 0.6]
ax_hist_right = fig2.add_axes(right_rect)
ax_hist_up = fig2.add_axes(up_rect)
ax_scatter = fig2.add_axes(main_rect)
ax_scatter.scatter(dens_Xt[:, 0], dens_Xt[:, 1], c='b')
ax_scatter.set_xlabel('principal #1')
ax_scatter.set_ylabel('principal #2')
_, right_bins_edges, _ = ax_hist_right.hist(dens_Xt[:, 1], 20, orientation='horizontal')
_, up_bins_edges, _ = ax_hist_up.hist(dens_Xt[:, 0], 20)
r_bin_width = right_bins_edges[1] - right_bins_edges[0]
u_bin_width = up_bins_edges[1] - up_bins_edges[0]
ax_hist_right.yaxis.set_major_formatter(NullFormatter())
ax_hist_up.xaxis.set_major_formatter(NullFormatter())

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

## range
fig4 = plt.figure(4)
gds = ImageGrid(fig4, 111, nrows_ncols=(3, 3), share_all=True)
n_plot = 0
for i in range(3):
    for j in range(4, 7):
        gds[n_plot].scatter(dens_Xt[:, i], dens_Xt[:, j], c='b')
        n_plot += 1
plt.show()

