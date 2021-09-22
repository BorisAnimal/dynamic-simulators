import matplotlib.pyplot as plt
import numpy as np

from impl.utils.lift import lift
from impl.utils.rk import rk
from src.controller import SquareWaveController
from src.plants.async_discrete_simulator import DiscreteAsyncSimulator
from src.runner import Runner
from vanderpol_plant import VanderpolAsyncPlant, VanderpolDynamic


# def df(t, x, u):
#     x1, x2 = x
#     return np.array([
#         2 * x2,
#         2 * x2 - 0.8 * x1 - 10 * x1 ** 2 * x2 - u
#     ])


def generate_data(dof, Nsteps, Nsim):
    """
    # Nsim # количество симуляций
    # Nsteps # количество шагов внутри симуляции
    """
    Ubig = np.random.rand(Nsteps, Nsim) * 2 - 1
    Xcurrent = np.random.rand(dof, Nsim) * 2 - 1
    df = VanderpolDynamic()
    X, Y, U = [], [], []
    for i in range(Nsteps):
        Xnext = rk(df, 0, Xcurrent, Ubig[i, :])
        X.append(Xcurrent)
        Y.append(Xnext)
        U.append(Ubig[i, :])
        Xcurrent = Xnext

    X = np.hstack(X)
    Y = np.hstack(Y)
    U = np.hstack(U).reshape((1, -1))
    return X, Y, U


if __name__ == '__main__':
    # Collect data
    X, Y, U = generate_data(dof=2, Nsteps=400, Nsim=1000)

    print("Data shapes:")
    print("X:", X.shape)
    print("Y:", Y.shape)
    print("U:", U.shape)

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

    # Predictor comparison
    Tmax = 3
    x0 = [0.5, 0.5]
    x0_lift = lift(x0)

    plant_true = VanderpolAsyncPlant(x0=x0, t=Tmax, data_file="vdp_true.dat")
    plant_koopman = DiscreteAsyncSimulator(x0=x0, t=Tmax, A=Alift, B=Blift, C=Clift, lift=lift,
                                           data_file="vdp_koopman.dat")
    controller = SquareWaveController(A=1.0, per=0.3)
    for plant in [plant_true, plant_koopman]:
        Runner(plant, controller).run()
        fields = plant.load_history()
        if plant == plant_true:
            data = fields['xs_desc']
        else:
            data = fields['xs']
        # Plot data
        x1, x2 = data[:, 0], data[:, 1]
        t = lambda x: np.linspace(0, plant.t.value, len(x))
        if plant == plant_true:
            plt.plot(t(x1), x1, label='x1')
            plt.plot(t(x2), x2, label='x2')
        else:
            plt.plot(t(x1), x1, '--', label='x1_k')
            plt.plot(t(x2), x2, '--', label='x2_k')
    plt.legend()
    plt.show()
