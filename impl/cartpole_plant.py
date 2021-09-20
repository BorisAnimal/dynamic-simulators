import numpy as np

from src.controller import ConstantController
from src.plants.async_simulator import AsyncSimulator
from src.runner import Runner


class CartpoleAsyncPlant(AsyncSimulator):
    def __init__(self, x0, g=9.81, l=1.0, b=0.1, mp=1.0, mc=2.0,
                 data_file='cartpole_results.pkl', **kwargs):
        super().__init__(x0=x0, data_file=data_file, **kwargs)
        # specified params
        self.g = g
        self.l = l
        self.b = b
        self.mp = mp
        self.mc = mc

    def dynamic(self, state, t, u):
        [theta, dtheta, x, dx] = state
        g, l, b, mp, mc = self.g, self.l, self.b, self.mp, self.mc
        S, C = np.sin(theta), np.cos(theta)

        ddx = (u + mp * S * (l * dtheta ** 2 + g * C)) / (mc + mp * S ** 2)
        ddtheta = -1 * (u * C + mp * l * dtheta ** 2 * C * S + (mc + mp) * g * S) / (l * (mc + mp * S ** 2))

        return dtheta, ddtheta, dx, ddx


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Setup environment
    plant = CartpoleAsyncPlant(x0=[0, 1.0, 0, 0], t=5)
    controller = ConstantController(0.0)
    # Run simulation
    Runner(plant, controller).run()
    # Load history and prepare for plotting
    fields = plant.load_history()
    data = fields['xs_desc']
    ts = fields['ts']
    print(data.shape)
    theta, dtheta = data[:, 0], data[:, 1]
    x, dx = data[:, 2], data[:, 3]
    # Plot data
    plt.plot(np.linspace(0, 1, len(x)), x, label='x')
    plt.plot(np.linspace(0, 1, len(theta)), theta, label='theta')
    plt.legend()
    plt.show()
