# pair-wise distance of dataset

import numpy as np 
import pickle
import matplotlib.pyplot as plt  
from matplotlib.ticker import NullFormatter, FuncFormatter


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

with open('/Users/hongbinren/Documents/program/MLEK/Burke/quantumX1D', 'rb') as f:
    data = pickle.load(f)
dens = data[:, 2:]
n = dens.shape[0]

# pair_wise distance
pair_wise_D = np.zeros(n*(n-1))
for i in range(n):
    D_ = np.sqrt(np.sum((dens[i][np.newaxis, :] - np.delete(dens, i, axis=0))**2, axis=1))
    pair_wise_D[i*(n-1):(i+1)*(n-1)] = D_
pair_wise_D = np.unique(pair_wise_D)

hist_nums, bins_edge, _ = ax1.hist(pair_wise_D, bins=20, density=True)
bin_width = bins_edge[1] - bins_edge[0]
def topercent(y, position):
    pos = np.round((bin_width*y), 2)
    return pos
func_fmt = FuncFormatter(topercent)
ax1.yaxis.set_major_formatter(func_fmt)
ax1.set_xlabel('distance')
ax1.set_ylabel('fraction')

# spatial distribution
mean = np.mean(dens, axis=0)
Cov = (dens - mean).T @ (dens - mean) / n
U, S, _ = np.linalg.svd(Cov)
dens_Xt = (dens - mean) @ U[:, :2]

dens_Xt_1 = dens_Xt[data[:, 0]==1]
dens_Xt_2 = dens_Xt[data[:, 0]==2]
dens_Xt_3 = dens_Xt[data[:, 0]==3]
dens_Xt_4 = dens_Xt[data[:, 0]==4]

ax2.scatter(dens_Xt_1[:, 0], dens_Xt_1[:, 1], c='b', label='num=1')
ax2.scatter(dens_Xt_2[:, 0], dens_Xt_2[:, 1], c='g', label='num=2')
ax2.scatter(dens_Xt_3[:, 0], dens_Xt_3[:, 1], c='y', label='num=3')
ax2.scatter(dens_Xt_4[:, 0], dens_Xt_4[:, 1], c='r', label='num=4')
ax2.legend()
ax2.set_xlabel('principal dim #1')
ax2.set_ylabel('principal dim #2')

plt.figure(2, figsize=(5, 5))
nullfmt = NullFormatter()         # no labels
dens_ne1 = dens[data[:, 0]==1]
n_ne1 = dens_ne1.shape[0]
mean_ne1 = np.mean(dens_ne1, axis=0)
Cov_ne1 = (dens_ne1 - mean_ne1).T @ (dens_ne1 - mean_ne1) / n_ne1
U1, S1, _ = np.linalg.svd(Cov_ne1)
dens_ne1_Xt = (dens_ne1 - mean_ne1) @ U1[:, :2]
x, y = dens_ne1_Xt[:, 0], dens_ne1_Xt[:, 1]

# definitions for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
bottom_h = left_h = left + width + 0.02

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.2]
rect_histy = [left_h, bottom, 0.2, height]

axScatter = plt.axes(rect_scatter)
axHistx = plt.axes(rect_histx)
axHisty = plt.axes(rect_histy)

# no labels
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)

# the scatter plot:
axScatter.scatter(x, y)

# now determine nice limits by hand:
binwidth = 0.25
xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
lim = (int(xymax/binwidth) + 1) * binwidth

# axScatter.set_xlim((-lim, lim))
# axScatter.set_ylim((-lim, lim))
axScatter.set_xlabel('principal dim #1')
axScatter.set_ylabel('principal dim #2')

bins = np.arange(-lim, lim + binwidth, binwidth)
axHistx.hist(x, bins=bins)
axHisty.hist(y, bins=bins, orientation='horizontal')

axHistx.set_xlim(axScatter.get_xlim())
axHisty.set_ylim(axScatter.get_ylim())

plt.show()