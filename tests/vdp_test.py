from impl.VanDerPol.vanderpol_plant import VanderpolAsyncPlant
from src.controller import SquareWaveController
from tests.base_test import *


class TestPlantImpl(BaseTest):
    def test_vdp_plant_impl(self):
        ## Setup environment
        plant = VanderpolAsyncPlant(x0=[0.5, 0.5], t=t, between_steps=bs, control_hz=chz)
        controller = SquareWaveController(1, 0.3)
        # Run simulation
        self.async_plant_check(plant, controller, [0, 0])


if __name__ == '__main__':
    unittest.main()
