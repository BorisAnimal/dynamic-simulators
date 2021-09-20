import numpy as np

from src.plants.async_simulator import AsyncSimulator


class PendulumAsyncPlant(AsyncSimulator):
    def __init__(self, x0, g=9.81, l=1.0, b=0.1, m=1.0,
                 data_file='pendulum_data.pkl', **kwargs):
        super().__init__(x0=x0, data_file=data_file, **kwargs)
        # specified params
        self.g = g
        self.l = l
        self.b = b
        self.m = m

    def dynamic(self, state, t, u):
        [theta, dtheta] = state
        ddtheta = (u - self.b * dtheta - self.m * self.g * self.l * np.sin(
            theta)) / (self.m * self.l ** 2)

        return dtheta, ddtheta
