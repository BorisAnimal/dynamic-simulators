import numpy as np

from impl.pendubot.params import *
from impl.pendubot.pendubot import Pendubot
from src.controller import ConstantController
from src.plants.async_simulator import AsyncSimulator
from src.runner import Runner


class PendubotPlant(AsyncSimulator):
    def __init__(self, x0, t, pendubot: Pendubot, data_file='pendubot_results.pkl', **kwargs):
        super().__init__(x0=x0, t=t, data_file=data_file, **kwargs)
        # specified params
        self.pendubot = pendubot

    def dynamic(self, state, t, u):
        q1, dq1, q2, dq2 = state
        ddq = self.pendubot.ddq(state, u)
        return [dq1, ddq[0], dq2, ddq[1]]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ## Setup environment
    # pendubot = Pendubot(pendubot_params_no_fric)
    pendubot = Pendubot(pendubot_params_with_fric)
    plant = PendubotPlant(x0=[2, 0, -2, 0], t=30, pendubot=pendubot)
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
    plt.plot(t(x1), x1, label='x1')
    plt.plot(t(x2), x2, label='x2')
    plt.grid()
    # plt.plot(t(u), u)
    plt.legend()
    plt.show()
