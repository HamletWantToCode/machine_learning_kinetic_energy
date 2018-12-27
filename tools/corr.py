import pickle
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import ImageGrid

np.random.seed(4256942)
with open('../data_file/quantum', 'rb') as f:
    data = pickle.load(f)
dens_x, Ek = np.fft.irfft(data[:, 1:], 100, axis=1)*100, data[:, 0].real

mean_X = np.mean(dens_x, axis=0)
Cov = (dens_x - mean_X).T @ (dens_x - mean_X) / 2000
U, S, _ = np.linalg.svd(Cov)
dens_Xt = (dens_x - mean_X) @ U[:, :4]

fig = plt.figure(figsize=(6, 6))
gds = ImageGrid(fig, 111, nrows_ncols=(2, 2), share_all=True, aspect=False, axes_pad=0.5)
for i in range(4):
    gds[i].plot(dens_Xt[:, i], Ek, 'bo')
fig.text(0.5, 0.04, 'principal components', ha='center')
fig.text(0.04, 0.5, 'Ek', va='center', rotation='vertical')
plt.show()
