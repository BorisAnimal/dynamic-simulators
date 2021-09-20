import numpy as np


def lift(x, N_rbf=100):
    x = np.array(x).reshape((-1, 1))
    return np.vstack([
        x,
        _rbf(x, N=N_rbf)
    ])


def _rbf(x, N=100):
    d2 = len(x)
    x = np.array(x).reshape((1, d2))
    C = np.random.random((N, d2)) * 2 - 1
    norm = np.linalg.norm(C - x, 2, 1, True)
    res = norm ** 2 * np.log(norm)
    return res
