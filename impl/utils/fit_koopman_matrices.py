import numpy as np


def fit_koopman_matrices(X, Y, U, lift):
    # Lift data
    Xlift = lift(X)
    Ylift = lift(Y)
    Nlift = Xlift.shape[0]

    print("X lifted:", Xlift.shape)
    print("Y lifted:", Ylift.shape)

    # Predictor matrices computing
    W = np.vstack([Ylift, X])
    V = np.vstack([Xlift, U])
    VVt = V.dot(V.T)
    WVt = W.dot(V.T)
    M = WVt.dot(np.linalg.pinv(VVt))
    Alift = M[:Nlift, :Nlift]
    Blift = M[:Nlift, Nlift:]
    Clift = M[Nlift:, :Nlift]

    return Alift, Blift, Clift