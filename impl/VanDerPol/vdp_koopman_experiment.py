import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from impl.utils.lift import lift
from src.controller import SquareWaveController
from src.runner import Runner
from vanderpol_plant import VanderpolAsyncPlant

if __name__ == '__main__':
    # Collect data
    n = 2
    Nsim = 400  # количество симуляций
    Ntraj = 1000  # количество шагов внутри симуляции

    # Notion: assumed data [-1;1] - what if not?
    Ubig = 2 * np.random.rand(Nsim, Ntraj) - 1
    X0 = 4 * np.random.rand(n, Nsim) - 1

    X, Y, U = [], [], []
    plant_for_data = VanderpolAsyncPlant(x0=[0, 0], t=1, control_hz=100, between_steps=10)  # ~10 sec each simulation
    for i in tqdm(range(Nsim), "Prepare data"):
        line = []
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
    print("Data shapes:")
    print("X:", X.shape)
    print("Y:", Y.shape)
    print("U:", U.shape)

    Xlift = np.hstack([lift(X[:, i]) for i in range(X.shape[1])])
    Ylift = np.hstack([lift(Y[:, i]) for i in range(Y.shape[1])])
    Nlift = Xlift.shape[0]

    print("X lifted:", Xlift.shape)
    print("Y lifted:", Ylift.shape)

    # Predictor
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
    # plant_koopman = DiscreteAsyncSimulator(x0=x0, t=Tmax, A=Alift, B=Blift, C=Clift, lift=lift,
    #                                        data_file="vdp_koopman.dat")
    controller = SquareWaveController(A=1.0, per=0.3)
    for plant in [plant_true]:
        Runner(plant, controller).run()
        fields = plant.load_history()
        data = fields['xs_desc']
        # Plot data
        x1, x2 = data[:, 0], data[:, 1]
        t = lambda x: np.linspace(0, plant.t.value, len(x))
        plt.plot(t(x1), x1, label='x1')
        plt.plot(t(x2), x2, label='x2')
    # plant_koopman = KoopmanAsyncPlant(x0=x0, t=Tmax, A=Alift,
    # B=Blift, C=Clift, lift=lift,
    #                                   data_file="data/koopman.dat")
    # TODO:
    x_disc = []
    for i in range(Tmax * 100):
        ti = i / 100.0
        z = lift(x0)
        u = controller.control(x0, ti)
        z1 = Alift.dot(z) + Blift.dot(u)
        x1 = Clift.dot(z1)
        x_disc.append(x1.flatten())
        x0 = x1
    xk = np.vstack(x_disc)
    print(xk.shape)
    x1 = xk[:, 0]
    x2 = xk[:, 1]
    plt.plot(t(x1), x1, '--', label='x1_k')
    plt.plot(t(x2), x2, '--', label='x2_k')

    plt.legend()
    plt.show()
