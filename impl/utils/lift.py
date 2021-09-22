import numpy as np


def lift(x, N_rbf=100):
    """
        lift :: D*N -> (D+N_rbf)*N
    """
    x = np.array(x)
    if len(x.shape) == 1:
        x = x.reshape((-1, 1))
    return np.vstack([
        x,
        _rbf(x, N_rbf=N_rbf)
    ])


def _rbf(x, N_rbf=100):
    """
        _rbf :: D*N -> N_rbf*N
    """
    x = np.array(x)
    Cbig = np.random.RandomState(42).rand(N_rbf, x.shape[0])
    Y = []
    for i in range(N_rbf):
        C = Cbig[i].reshape(-1, 1)
        norm = np.linalg.norm(x - C, 2, 0, True)
        y = norm ** 2 * np.log(norm)
        y[np.isnan(y)] = 0.0
        Y.append(y)
    return np.vstack(Y)


if __name__ == '__main__':
    xl = lift(np.random.rand(2, 4000))
    print(xl.shape, 'expected:', (102, 4000))
