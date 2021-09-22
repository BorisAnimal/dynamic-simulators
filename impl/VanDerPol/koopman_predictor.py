import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from impl.utils.lift import lift
from src.controller import SquareWaveController
from src.plants.async_discrete_simulator import DiscreteAsyncSimulator
from src.runner import Runner
from vanderpol_plant import VanderpolAsyncPlant


def generate_data(dof, Nsim, Ntraj):
    """
    # Nsim # количество симуляций
    # Ntraj # количество шагов внутри симуляции
    """
    # Notion: assumed data [-1;1] - what if not?
    Ubig = 2 * np.random.rand(Nsim, Ntraj) - 1
    X0 = 4 * np.random.rand(dof, Nsim) - 1

    X, Y, U = [], [], []
    # TODO: попробовать через sync_plant
    plant_for_data = VanderpolAsyncPlant(x0=np.zeros((dof,)), t=1, control_hz=100,
                                         between_steps=4)  # ~10 sec each simulation
    for i in tqdm(range(Nsim), "Prepare data"):
        x0 = X0[:, i]  # 2x1
        us = Ubig[i]  # N
        for j in range(Ntraj):
            x1 = plant_for_data.step(x0, j, us[j])[-1]
            X.append(x0.reshape((-1, 1)))
            Y.append(x1.reshape(-1, 1))
            x0 = x1
        U.append(us.reshape(1, -1))

    X = np.hstack(X)
    Y = np.hstack(Y)
    U = np.hstack(U)
    return X, Y, U


if __name__ == '__main__':
    # Collect data
    X, Y, U = generate_data(dof=2, Nsim=400, Ntraj=1000)

    print("Data shapes:")
    print("X:", X.shape)
    print("Y:", Y.shape)
    print("U:", U.shape)

    # Lift data
    Xlift = np.hstack([lift(X[:, i]) for i in range(X.shape[1])])
    Ylift = np.hstack([lift(Y[:, i]) for i in range(Y.shape[1])])
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
