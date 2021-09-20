import sys

import numpy as np

from src.controller import SquareWaveController
from src.plants.async_simulator import AsyncSimulator
from src.runner import Runner


class DiscreteAsyncSimulator(AsyncSimulator):
    def __init__(self, x0, A, B, C, lift=lambda x: np.array(x).reshape((-1, 1)),
                 data_file="discrete_sim.pkl",
                 t=sys.maxsize, control_hz=100, between_steps=50):
        super().__init__(x0=x0, data_file=data_file, t=t, control_hz=control_hz, between_steps=between_steps)
        # specified params
        self.A = A
        self.B = B
        self.C = C
        self.lift = lift

    def dynamic(self, state, t, u):
        """
        param state: state_{k}
        return: state_{k+1}
        """
        z = np.array(self.lift(state)).reshape((-1, 1))
        state1 = self.A.dot(z) + self.B.dot(u)
        return self.C.dot(state1).flatten()

    def step(self, state, t, u):
        return self.dynamic(state, t, u)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ## Setup environment
    plant = DiscreteAsyncSimulator(
        x0=[0.5, 0.5], t=3, A=np.eye(3, 3), B=np.eye(3, 1), C=np.eye(2, 3),
        lift=lambda x: np.array([x[0], x[0] * 2, x[1] * 3]).reshape((-1, 1))
    )
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
