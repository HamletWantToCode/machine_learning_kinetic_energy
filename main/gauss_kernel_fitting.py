# loss function that learn function and it's derivative simutaneously

import numpy as np 
from statslib.main.base import BaseRegressor
from statslib.main.base import BaseOptimize
from statslib.main.optimization import *
from statslib.tools.utils import regularizer, regularizer_gradient

def special_load_data(data, n_batch=1, fst_end=None, snd_end=None):
    N = data.shape[0]
    n_elements = N//n_batch
    for i in range(n_batch):
        partition = data[i*n_elements : (i+1)*n_elements]
        sub_X, sub_dy, sub_KM, sub_y = partition[:, :fst_end], partition[:, fst_end:snd_end], partition[:, snd_end:-1], partition[:, -1]
        yield (sub_KM, sub_X, sub_y, sub_dy)

class NewLoss_KRR(BaseRegressor):
    def __init__(self, kernel, Lambda, beta, optimizer):
        super().__init__(kernel, Lambda, optimizer)
        self.beta_ = beta

    def lossFunction(self, alpha, sub_KM, sub_X, sub_y, sub_dy, gamma):
        n_samples = sub_KM.shape[0]
        delta_y = sub_y - (sub_KM @ alpha)
        loss_on_function = 0.5*(delta_y @ delta_y)
        loss_on_gradient = 0 
        for i in range(n_samples):
            delta_X = (sub_X[i][np.newaxis, :]) - self.X_fit_
            estimate_gradient = -2*gamma*((alpha * sub_KM[i]) @ delta_X)
            loss_on_gradient += ((sub_dy[i] - estimate_gradient).conj() @ (sub_dy[i] - estimate_gradient)).real
        loss_on_gradient *= 0.5
        regular_term = self.regular_func(alpha)
        return (1.0/n_samples)*(loss_on_function + self.beta_*loss_on_gradient) + 0.5*self.Lambda_*regular_term

    def lossGradient(self, alpha, sub_KM, sub_X, sub_y, sub_dy, gamma):
        n_samples = sub_KM.shape[0]
        delta_y = sub_y - (sub_KM @ alpha)
        derivative_on_function = -1*(delta_y @ sub_KM)
        derivative_on_gradient = 0
        for i in range(n_samples):
            delta_X = (sub_X[i][np.newaxis, :]) - self.X_fit_
            estimate_gradient = -2*gamma*((alpha * sub_KM[i]) @ delta_X )
            function_hessan = -2*gamma*(delta_X.T)*(sub_KM[i][np.newaxis, :])
            derivative_on_gradient += (sub_dy[i] - estimate_gradient).conj() @ function_hessan
        derivative_on_gradient *= -1
        regular_grad_term = self.regular_grad(alpha)
        return (1.0/n_samples)*(derivative_on_function + self.beta_*derivative_on_gradient) + self.Lambda_*regular_grad_term

    def fit(self, X, y, dy, gamma):
        n_sample = X.shape[0]
        self.X_fit_ = X
        full_KM = self.kernel(X)
        self.regular_func = regularizer(full_KM)
        self.regular_grad = regularizer_gradient(full_KM)
        if self.optimizer.alpha0_ is not None:
            alpha0 = self.optimizer.alpha0_
        else:
            alpha0 = np.random.uniform(-1, 1, n_sample)*1e-2
        alpha = self.optimizer.run(alpha0, self.lossFunction, self.lossGradient, full_KM, X, y, dy, gamma)
        self.coef_ = alpha
        return self

class Special_optimizer(object):
    def run(self, alpha, function, gradient, full_KM, X, y, dy, gamma):
            data = np.c_[X, dy, full_KM, y]
            f0 = function(alpha, full_KM, X, y, dy, gamma)
            n_epoch = 0
            while True:
                loader = special_load_data(data, self.nb_, 31, 62)
                for i, (sub_KM, sub_X, sub_y, sub_dy) in enumerate(loader):
                    alpha = self.optimizer(alpha, gradient, sub_KM, sub_X, sub_y, sub_dy, gamma)
                    if self.verbose_:
                        f_update = function(alpha, full_KM, X, y, dy, gamma)
                        self.fvals_.append(f_update)
                f1 = function(alpha, full_KM, X, y, dy, gamma)
                ferr = abs(f1-f0)
                if ferr < self.stopError_:
                    print('optimization converges after %d epoches and %d batch iterations!' %(n_epoch, i+1))
                    break
                f0 = f1
                if n_epoch > self.maxiters_:
                    raise StopIteration('loop exceeds the maximum iterations !')
                n_epoch += 1
                if n_epoch > 50:
                    self.lr_ *= 0.99
            return alpha

class Special_SGD(Special_optimizer, GradientDescent):
    def __init__(self, learning_rate, stopError, maxiters, alpha0=None, **kwargs):
        super().__init__(learning_rate, stopError, maxiters, **kwargs)
        if alpha0 is not None:
            self.alpha0_ = alpha0

class Special_momentum(Special_optimizer, NesterovGD):
    def __init__(self, learning_rate, stopError, maxiters, momentum_param, alpha0=None, **kwargs):
        super().__init__(learning_rate, stopError, maxiters, momentum_param, **kwargs)
        if alpha0 is not None:
            self.alpha0_ = alpha0

# if __name__ == '__main__':
#     import pickle
#     fname = '/Users/hongbinren/Downloads/mnt/project/ML_periodic/quantum1D'
#     with open(fname, 'rb') as f:
#         fst_principal = pickle.load(f)
#     fname1 = '/Users/hongbinren/Downloads/mnt/project/ML_periodic/potential1D'
#     with open(fname1, 'rb') as f1:
#         potential = pickle.load(f1)
#     n_samples = fst_principal.shape[0]
#     index = np.arange(0, n_samples, 1, dtype=int)
#     np.random.shuffle(index)

#     density, kinetic_energy = fst_principal[index[:100], :-1], fst_principal[index[:100], -1]
#     external_potential = potential[index[:100], 1:]                        
#     # the fourier components of potential only 
#     # preserve positive frequency part, negative
#     # part is the same, and we regenerate it from the following code
#     external_potential = np.c_[external_potential, external_potential[:, ::-1][:, :-1]]    
#     chemical_potential = np.zeros_like(external_potential)
#     chemical_potential[:, 0] = potential[index[:100], 0]
#     total_potential = external_potential - chemical_potential

#     def numerical_check(x, f, df, h, *args):
#         f0 = f(x, *args)
#         gradient = df(x, *args)
#         x_ahead = x + h*gradient
#         f_ahead = f(x_ahead, *args)
#         f_grad = f0 + h*(gradient @ gradient)
#         return abs(f_ahead - f_grad)/f_ahead, np.sqrt((h**2)*(gradient @ gradient))

#     from statslib.tools.utils import rbfKernel, regularizer, regularizer_gradient
#     gamma = 0.1
#     rbf = rbfKernel(gamma)
#     model = NewLoss_KRR(rbf, 0, None)
#     full_KM = rbf(density)
#     model.regular_func = regularizer(full_KM)
#     model.regular_grad = regularizer_gradient(full_KM)
#     model.X_fit_ = density
#     f = model.lossFunction
#     df = model.lossGradient
#     alpha = np.random.uniform(-10, 10, 100)
#     err = numerical_check(alpha, f, df, 1e-3, full_KM, density, kinetic_energy, total_potential, gamma)
#     print(err)