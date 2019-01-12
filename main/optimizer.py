import numpy as np 
from sklearn.utils import check_array

class Minimizer(object):
    def __init__(self, ml_model):
        self.ml_model = ml_model

    def energy(self, denst, Vxt, mu, N):
        Ek = self.ml_model.predict(denst)
        return Ek + np.sum((Vxt - mu)*denst) + mu*N 

    def energy_gd(self, denst, Vxt, mu):
        dEk = self.ml_model.predict_gradient(denst)
        return dEk + (Vxt - mu)

    def run(self, densx_init, Vx, mu, N, eta=0.1, err=1e-3, maxiters=1000):
        dens_init = check_array(dens_init)
        denst_init = self.ml_model.transform(dens_init)
        Vxt = self.ml_model.transform_gradient(Vx)

        E0 = self.energy(denst_init, Vxt, mu, N)
        n = 1
        while True:
            gd = self.energy_gd(denst_init, Vxt, mu)
            denst = denst_init - eta*gd
            E1 = self.energy(denst, Vxt, mu, N)
            denst_init = denst
            if abs(E1 - E0)<err:
                print('converge after %d of iterations !' %(n))
                break
            E0 = E1
            n += 1
            if n > maxiters:
                raise StopIteration('exceed maximum iteration number !')
        dens_optim = self.ml_model.inverse_transform(denst_init)
        return dens_optim