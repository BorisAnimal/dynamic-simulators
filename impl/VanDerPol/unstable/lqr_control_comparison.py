import matplotlib.pyplot as plt
import numpy as np

from impl.utils.fit_koopman_matrices import fit_koopman_matrices
from impl.utils.generate_data import generate_data
from impl.utils.lift import lift
from src.controller import LQR
from src.plants.async_discrete_simulator import DiscreteAsyncSimulator
from src.runner import Runner
from ..vanderpol_plant import VanderpolAsyncPlant, VanderpolDynamic

"""
TODO:
    + Define system
    + Apply Koopman fitting
    + Linearize vdp system
    Run stabilization simulation task on both: Koopman and original system (linearized tbh)
    Compare the area of attraction for both (Koopman must be >= original)
"""

if __name__ == '__main__':
    ## Predictor comparison
    Tmax = 4
    x0 = [0.5, 0.0]
    x0_sim = [0.45, 0.05]
    x0_lift = lift(x0)

    dynamic = VanderpolDynamic(stable_zero=False)

    ## Koopman staff
    # Collect data
    X, Y, U = generate_data(dynamic, dof=2, Nsteps=400, Nsim=1000)
    print("Data shapes:")
    print("X:", X.shape)
    print("Y:", Y.shape)
    print("U:", U.shape)
    Alift, Blift, Clift = fit_koopman_matrices(X, Y, U, lift)
    # u0_lift =
    controller = LQR(x0, Alift, Blift, u0=0.0, Q=np.diag([5, 1]), R=np.diag([0.1]), Umax=0.5, lift=lift)

    ## Linearized staff
    A0, B0, u0 = dynamic.vdp_lin(np.array(x0).reshape((-1, 1)))

    # controller = LQR(x0, A0, B0, u0, Q=np.diag([5, 1]), R=np.diag([0.1]), Umax=0.5)
    # controller = ConstantController(0.0)

    plant_true = VanderpolAsyncPlant(x0=x0_sim, t=Tmax, data_file="vdp_true.dat")
    plant_koopman = DiscreteAsyncSimulator(x0=x0_sim, t=Tmax, A=Alift, B=Blift, C=Clift, lift=lift,
                                           data_file="vdp_koopman.dat")
    # controller = SquareWaveController(A=1.0, per=0.3)

    # for plant in [plant_true, plant_koopman]:
    for plant in [plant_koopman]:
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
    plt.ylim([-2, 2])
    plt.show()
