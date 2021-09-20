import numpy as np

from impl.cartpole_plant import CartpoleAsyncPlant
from impl.pendulum_plant import PendulumAsyncPlant
from src.controller import PDController, ConstantController
from src.trajectory import harmonic_trajectory_builder
from tests.base_test import *


class TestPlantImpl(BaseTest):
    def test_pendulum_plant_impl(self):
        # Setup environment
        plant = PendulumAsyncPlant(x0=[0, 0], t=t, control_hz=chz, between_steps=bs)
        trajectory = harmonic_trajectory_builder(A0=0, A=np.pi / 6, freq=1)
        controller = PDController(trajectory=trajectory, kp=50, kd=40)
        self.async_plant_check(plant, controller, [0, 0])

    def test_cartpole_plant_impl(self):
        # Setup environment
        plant = CartpoleAsyncPlant(x0=[0, 1.0, 0, 0], t=t, between_steps=bs, control_hz=chz)
        controller = ConstantController(0.0)
        self.async_plant_check(plant, controller, [0, 0, 0, 0])


if __name__ == '__main__':
    unittest.main()
