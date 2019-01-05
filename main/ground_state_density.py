import numpy as np 
from statslib.tools.utils import rbfKernel, rbfKernel_gd

class Ground_state_density(object):
    def __init__(self, gamma, alpha, train_X, n_cmp):
        self.alpha_ = alpha
        self.kernel = rbfKernel(gamma)
        self.kernel_gd = rbfKernel_gd(gamma)
        mean_x = np.mean(train_X, axis=0, keepdims=True)
        n = train_X.shape[0]
        Cov = (train_X - mean_x).T @ (train_X - mean_x) / n
        U, _, _ = np.linalg.svd(Cov)
        self.mean_ = mean_x
        self.tr_mat_ = U[:, :n_cmp]
        self.Xt_fit_ = (train_X - mean_x) @ U[:, :n_cmp]

    def energy(self, dens, Vx, mu, N):
        _, D_ = dens.shape
        dens_t = (dens - self.mean_) @ self.tr_mat_
        Ek = (self.kernel(dens_t, self.Xt_fit_) @ self.alpha_)[0]
        return Ek + np.sum((Vx - mu)*dens)*(1.0/(D_-1)) + mu*N 

    def energy_gd(self, dens, Vx, mu):
        N_, D_ = dens.shape
        dens_t = (dens - self.mean_) @ self.tr_mat_
        dEk = (D_-1)*(self.kernel_gd(dens_t, self.Xt_fit_) @ self.alpha_).reshape((N_, -1))
        shifted_Vx = Vx - mu
        project_gd = (dEk + shifted_Vx @ self.tr_mat_) @ self.tr_mat_.T
        return project_gd

    def optimize(self, dens_init, Vx, mu, N, eta, err, maxiters=1000):
        assert dens_init.ndim == 2, print('dimension mismatch')
        E0 = self.energy(dens_init, Vx, mu, N)
        n = 1
        while True:
            gd = self.energy_gd(dens_init, Vx, mu)
            dens = dens_init - eta*gd
            E1 = self.energy(dens, Vx, mu, N)
            dens_init = dens
            if abs(E1 - E0)<err:
                print('converge after %d of iterations !' %(n))
                break
            E0 = E1
            n += 1
            if n > maxiters:
                raise StopIteration('exceed maximum iteration number !')
        return dens_init[0]