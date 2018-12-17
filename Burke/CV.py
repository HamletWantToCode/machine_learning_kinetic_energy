import pickle
import numpy as np
from mpi4py import MPI
from statslib.main.kernel_ridge import KernelRidge
from statslib.tools.utils import rbfKernel, meanSquareError
from statslib.main.cross_validation import Cross_validation
from sklearn.model_selection import StratifiedKFold

np.random.seed(8)

with open('quantumX1D', 'rb') as f:
    data = pickle.load(f)
np.random.shuffle(data)
train_data = data[:1001]
test_data = data[1001:]

train_single = train_data[train_data[:, 0]==1]

train_X, train_y = train_single[:200, 2:], train_single[:200, 1]
labels = train_single[:200, 0]
skr = StratifiedKFold(5, True, 7)

# plot coef contour
n_s, n_l = 50, 20
Sigma = np.logspace(-2, 3, n_s)
Lambda = np.logspace(-16, -4, n_l)
xx, yy = np.meshgrid(Sigma, Lambda)
XY = np.c_[xx.reshape((-1, 1)), yy.reshape((-1, 1))]
Z = None

comm = MPI.COMM_WORLD
n_cpu = comm.Get_size()
ID = comm.Get_rank()
Elements = len(XY)//n_cpu
part_XY = XY[ID*Elements:(ID+1)*Elements]
part_Z = np.zeros(Elements)

for i, (sigma, lambda_) in enumerate(part_XY):
    gamma = 1.0/(2*sigma**2)
    kernel = rbfKernel(gamma)
    model = KernelRidge(kernel, lambda_)
    sp = skr.split(train_X, labels)
    CV = Cross_validation(sp, model, meanSquareError)
    avg_err = CV.run(train_X, train_y[:, np.newaxis])
    part_Z[i] = avg_err

if ID == 0:
    Z = np.zeros(n_s*n_l)

comm.Gather(part_Z, Z, 0)

if ID == 0:
    surf_data = np.c_[XY, Z.reshape((-1, 1))]
    with open('error_surf', 'wb') as f:
        pickle.dump(surf_data, f)

