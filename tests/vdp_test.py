import unittest

from impl.VanDerPol.vanderpol_plant import VanderpolAsyncPlant
from src.controller import SquareWaveController
from src.runner import Runner

bs, chz, t = 10, 100, 1
N = t * bs * chz
N_desc = t * chz


class TestPlantImpl(unittest.TestCase):
    def test_vdp_plant_impl(self):
        ## Setup environment
        plant = VanderpolAsyncPlant(x0=[0.5, 0.5], t=t, between_steps=bs, control_hz=chz)
        controller = SquareWaveController(1, 0.3)
        # Run simulation
        Runner(plant, controller).run()
        # Load history and prepare for plotting
        fields = plant.load_history()
        xs, xs_desc, ts, us = fields['xs'], fields['xs_desc'], fields['ts'], fields['us']
        self.assertEqual(ts.shape, (N_desc,))
        self.assertEqual(xs.shape, (N, 2))
        self.assertEqual(xs_desc.shape, (N_desc, 2))
        self.assertEqual(us.shape, (N_desc,))
        # Step works
        x1 = plant.step([0, 0], 0, 0)
        self.assertEqual(x1.shape, (bs, 2))


if __name__ == '__main__':
    unittest.main()
