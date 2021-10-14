import numpy as np

from impl.utils.rk import rk


def generate_data(dynamic, dof, Nsteps, Nsim):
    """
    # Nsim # количество симуляций
    # Nsteps # количество шагов внутри симуляции
    """
    Ubig = np.random.rand(Nsteps, Nsim) * 2 - 1
    Xcurrent = np.random.rand(dof, Nsim) * 2 - 1

    X, Y, U = [], [], []
    for i in range(Nsteps):
        Xnext = rk(dynamic, 0, Xcurrent, Ubig[i, :])
        X.append(Xcurrent)
        Y.append(Xnext)
        U.append(Ubig[i, :])
        Xcurrent = Xnext

    X = np.hstack(X)
    Y = np.hstack(Y)
    U = np.hstack(U).reshape((1, -1))
    return X, Y, U
