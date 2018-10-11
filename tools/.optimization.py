import pickle
import numpy as np 
from scipy.optimize import minimize
from QML.main import kinetic_energy_functional as Ekf
from model import model

def loss(alpha, c, gamma, Xi, trainX, trainY, traindY):
    nf, N = Xi.shape
    m, nf1 = trainX.shape
    assert len(alpha) == N, ('alpha should be arranged to (N)')
    assert nf1 == nf, ('shape of the matrix do not match')
    
    Y_pred = Ekf.kinetic_dens(trainX, alpha, Xi, gamma)
    assert len(Y_pred) == m
    L1 = np.linalg.norm(Y_pred - trainY)**2
    assert isinstance(L1, np.float64)

    dY_pred = Ekf.kinetic_deriv_dens(trainX, alpha, Xi, gamma)
    assert dY_pred.shape == (m, nf)
    L2 = np.sum(np.linalg.norm((dY_pred - traindY), axis=1)**2) 
    assert isinstance(L2, np.float64)

    return 0.5*(L1 + c*L2)    # float

def loss_deriv(alpha, c, gamma, Xi, trainX, trainY, traindY):
    nf, N = Xi.shape
    m, nf1 = trainX.shape
    assert len(alpha) == N, ('alpha should be arranged to (N)')
    assert nf1 == nf, ('shape of the matrix do not match')

    dYda = Ekf.kinetic_deriv_alpha(trainX, Xi, gamma)
    assert dYda.shape == (N, m)
    Y_pred = Ekf.kinetic_dens(trainX, alpha, Xi, gamma)
    assert len(Y_pred) == m
    dl1 = dYda.T @ (Y_pred - trainY)   # N*m*m = N
    assert len(dl1) == N

    kernel = 3*(gamma**3)*(trainX @ Xi)**2     # m*N
    assert kernel.shape == (m, N)
    dfx_da = np.zeros((m, nf, N))         # m*f*N
    for i, ki in enumerate(kernel):
        A = np.repeat(ki.reshape(1, -1), repeats=nf, axis=0)     # f*N
        dfx_da[i] = A*Xi                                # f*N
    dfx_da = np.transpose(dfx_da, axes=(1, 0, 2))      # f*m*N
    dY_pred = Ekf.kinetic_deriv_dens(trainX, alpha, Xi, gamma)
    assert dY_pred.shape == (m, nf)
    dl2 = np.squeeze(np.einsum('ij,jik->k', (dY_pred - traindY), dfx_da))    # 1*N
    assert len(dl2) == N

    return (dl1 + c*dl2).real

if __name__ == '__main__':
    fname = '../densG_T_dT'
    gamma = 0.1
    C_ = 1e-4

    KRR = model(fname, gamma, C_)
    with open('train_data', 'rb') as f:
        train_data = pickle.load(f)
    train_dens, train_Ek, train_dEk = train_data[:, :51], train_data[:, 51], train_data[:, 52:]

    alpha_init = KRR.dual_coef_
    Xi = KRR.X_fit_.T
    c = 1

    print('...start optimization')
    result = minimize(loss, alpha_init,
                        args=(c, gamma, Xi, train_dens, train_Ek, train_dEk),
                        method='Newton-CG', jac=loss_deriv, options={'eps': 1e-2})
    print(result.success)
    alpha_optim = result.x
    with open('../optimAlpha', 'wb') as f1:
        pickle.dump(alpha_optim, f1)





