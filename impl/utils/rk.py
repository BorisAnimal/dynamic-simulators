import numpy as np


def rk(df, t, x, u=None, dt=0.01):
    """
    Parameters
    ----------
        df :: (x:N, t, u) -> dx:N - derivative function
        t : float - time
        x : N - state
        u : float - control
        dt : float - time step
    """
    x = np.array(x)
    k1 = df(x, t, u)
    k2 = df(x + k1 * dt / 2, t, u)
    k3 = df(x + k2 * dt / 2, t, u)
    k4 = df(x + k1 * dt, t, u)

    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
