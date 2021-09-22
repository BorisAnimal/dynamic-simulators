import numpy as np

from impl.pendulum_plant import PendulumAsyncPlant
from src.controller import SquareWaveController, PDController, ConstantController
from src.plants.async_discrete_simulator import DiscreteAsyncSimulator
from src.plants.sync_plant import simulator
from src.trajectory import harmonic_trajectory_builder
from tests.base_test import *


class TestPlants(BaseTest):
    def test_async_discrete_simulator(self):
        ## Setup environment
        plant = DiscreteAsyncSimulator(
            x0=[0.5, 0.5], t=t, A=np.eye(3, 3), B=np.eye(3, 1), C=np.eye(2, 3),
            lift=lambda x: np.array([x[0], x[0] * 2, x[1] * 3]).reshape((-1, 1)),
            between_steps=bs, control_hz=chz,
        )
        controller = SquareWaveController(1, 0.3)
        # Run simulation
        self.async_discrete_plant_check(plant, controller, [0, 0])

    def test_async_simulator(self):
        # Setup environment
        plant = PendulumAsyncPlant(x0=[0, 0], t=t, control_hz=chz, between_steps=bs)
        trajectory = harmonic_trajectory_builder(A0=0, A=np.pi / 6, freq=1)
        controller = PDController(trajectory=trajectory, kp=50, kd=40)
        # Run simulation
        self.async_plant_check(plant, controller, [0, 0])

    def test_sync_plant(self):
        dynamic = lambda x, t, u, params: [
            x[1],
            (u - params['b'] * x[1] - params['m'] * params['g'] * params['l'] * np.sin(
                x[0])) / (params['m'] * params['l'] ** 2)
        ]
        control = ConstantController(0.2).control
        sys_params = {
            'm': 1,
            'g': 9.81,
            'l': 1,
            'b': 0.1,
        }

        x = simulator(dynamic, control, [0, 0], sys_params, t=t, between_steps=bs, control_hz=chz)
        self.assertEqual(x.shape, (N, 2))


if __name__ == '__main__':
    unittest.main()
