#!/usr/bin/env python

import numpy as np

def kernelFitting(X, y, penalty, kernelFunction):
    n = len(X)
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] = kernelFunction(X[i], X[i])
        for j in range(i+1, n):
            K[i, j] = K[j, i] = kernelFunction(X[i], X[j])
    K += penalty*n*np.eye(n)
    U, S, Vh = np.linalg.svd(K)
    rank = len(S[S>1e-10])
    coef = np.divide(np.dot(y, U), S)
    sol = 0
    for j, cj in enumerate(coef):
        sol += cj*Vh[j, :]
    return sol

def kernelRepresentedFunction(X, coef, kernelFunction):
    def func(x):
        n = len(coef)
        fvalue = 0
        for i in range(n):
            fvalue += coef[i]*kernelFunction(x, X[i])
        return fvalue
    return func


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sklearn.kernel_ridge import KernelRidge

    X = np.linspace(0, 1, 20)
    a, b, c, d = np.random.uniform(-10, 10, 4)
    f = lambda x: a + b*x + c*x**2 + d*x**3
    y = f(X)
    penalty = 1e-3
    kernel = lambda x, y: (1 + 100*x*y)**3
    coef = kernelFitting(X, y, penalty, kernel)
    func = kernelRepresentedFunction(X, coef, kernel)
    krr = KernelRidge(alpha=1e-3, kernel=kernel)
    krr.fit(X.reshape(-1, 1), y.reshape(-1, 1))

    X_test = np.linspace(1, 2, 30)
    y_test = f(X_test)
    y_pred = np.array([func(x) for x in X_test])
    y_sklearn = krr.predict(X_test.reshape(-1, 1))
    plt.plot(X_test, y_test, 'ro')
    plt.plot(X_test, y_pred, 'b-s')
    plt.plot(X_test, y_sklearn, 'g-*')
    plt.show()
