import numpy as np
from scipy.linalg import solve_continuous_are as are
from scipy.linalg import solve_discrete_are as are_disc

# https://colab.research.google.com/drive/1_MjEdyWbJ2In3NZRfdrmKUPdR0IobKH-#scrollTo=Z622gBBBxQzt
def lqr(A, B, Q, R, discr=False):
    # Solve the ARE
    if discr:
        S = are_disc(A, B, Q, R)
    else:
        S = are(A, B, Q, R)
    print(S)
    R_inv = np.linalg.inv(R)
    if discr:
        K = np.linalg.multi_dot([
            np.linalg.pinv(R + np.linalg.multi_dot([B.T, S, B])),
            B.T, S, A
        ])
    else:
        K = R_inv.dot((B.T).dot(S))  # R^{-1}B^{T}S
    Ac = A - B.dot(K)
    E = np.linalg.eigvals(Ac)
    return S, K, E  # Mostly K useful only
