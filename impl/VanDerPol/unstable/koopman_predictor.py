import matplotlib.pyplot as plt
import numpy as np

from impl.VanDerPol.vanderpol_plant import VanderpolDynamic, VanderpolAsyncPlant
from impl.utils.fit_koopman_matrices import fit_koopman_matrices
from impl.utils.generate_data import generate_data
from impl.utils.lift import lift
from src.controller import SquareWaveController
from src.plants.async_discrete_simulator import DiscreteAsyncSimulator
from src.runner import Runner

if __name__ == '__main__':
    # Collect data
    X, Y, U = generate_data(VanderpolDynamic(stable_zero=False), dof=2, Nsteps=400, Nsim=1000)
    U *= 0.001

    print("Data shapes:")
    print("X:", X.shape)
    print("Y:", Y.shape)
    print("U:", U.shape)

    Alift, Blift, Clift = fit_koopman_matrices(X, Y, U, lift)

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
