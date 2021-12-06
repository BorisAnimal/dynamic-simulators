import numpy as np
import warnings

warnings.filterwarnings('ignore')

import matplotlib.cm as cm

import matplotlib.pyplot as plt

from impl.pendubot.params import pendubot_params_with_fric
from impl.pendubot.pendubot import Pendubot
from impl.pendubot.pendubot_plant import PendubotPlant
from src.controller import ConstantController
from src.runner import Runner

if __name__ == '__main__':
    ## Setup environment
    # pendubot = Pendubot(pendubot_params_no_fric)
    pendubot = Pendubot(pendubot_params_with_fric)
    plant = PendubotPlant(x0=[2, 0, -2, 0], t=10, pendubot=pendubot)
    # plant = CartpoleAsyncPlant(t0=[-0.1, -0.5], t=3)
    controller = ConstantController(0.0)
    # controller = SquareWaveController(1, 0.3)
    # Run simulation
    Runner(plant, controller).run()
    # Load history and prepare for plotting
    fields = plant.load_history()
    data = fields['xs_desc']
    ts = fields['ts']
    print("Result data shape:", data.shape)
    x1, x2 = data[:, 0], data[:, 1]
    u = fields['us']
    # Plot data
    t = lambda x: np.linspace(0, plant.t.value, len(x))
    plt.plot(t(x1), x1, label='q1')
    plt.plot(t(x2), x2, label='q2')
    plt.grid()
    plt.legend()
    plt.show()

    import pykoopman as pk
    from pykoopman.common import advance_linear_system

    n_states = 4
    n_controls = 2
    n_measurements = len(data) - 1
    X = data[:-1]
    Y = data[1:]

    model = pk.Koopman()
    model.fit(data)
    K = model.koopman_matrix.real
    print(K)
    plt.matshow(K)
    plt.show()

    f = data
    f_predicted = np.vstack((f[0], model.simulate(f[0], n_steps=f.shape[0] - 1)))

    x1, x2 = f_predicted[:, 0], f_predicted[:, 1]
    plt.plot(t(x1), x1, label='q1')
    plt.plot(t(x2), x2, label='q2')
    plt.grid()
    plt.legend()
    plt.show()
