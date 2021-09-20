import numpy as np

from src.controller import SquareWaveController
from src.plants.async_simulator import AsyncSimulator
from src.runner import Runner


class VanderpolAsyncPlant(AsyncSimulator):
    def __init__(self, x0, t, data_file='vanderpol_results.pkl', **kwargs):
        super().__init__(x0=x0, t=t, data_file=data_file, **kwargs)
        # specified params

    def dynamic(self, state, t, u):
        [x1, x2] = state
        dx1 = 2 * x2
        dx2 = -0.8 * x1 + 2 * x2 - 10 * x1 ** 2 * x2 - u
        return dx1, dx2


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ## Setup environment
    plant = VanderpolAsyncPlant(x0=[0.5, 0.5], t=3)
    # plant = CartpoleAsyncPlant(x0=[-0.1, -0.5], t=3)
    controller = SquareWaveController(1, 0.3)
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
    # plt.plot(t(u), u)
    plt.legend()
    plt.show()
