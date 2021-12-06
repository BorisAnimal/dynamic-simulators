import matplotlib.pyplot as plt
import numpy as np

from impl.utils.generate_data import generate_data
from impl.utils.lift import lift
from src.controller import SquareWaveController
from src.plants.async_discrete_simulator import DiscreteAsyncSimulator
from src.runner import Runner
from tsa_lin_plant import TSALinAsyncPlant, TSALinDynamic

if __name__ == '__main__':
    # Collect data
    X, Y, U = generate_data(TSALinDynamic(), dof=2, Nsteps=400, Nsim=1000)

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
    x0 = [0.5, 200]
    x0_lift = lift(x0)

    plant_true = TSALinAsyncPlant(x0=x0, t=Tmax, data_file="vdp_true.dat")
    plant_true._dynamic.I *= 1.1
    plant_true._dynamic.b_x *= 0.9
    plant_true._dynamic.m *= 1.05
    plant_true._dynamic.r0 *= 1.1
    plant_true._dynamic.L0 *= 0.96
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