import unittest
from typing import Sized

from src.controller import Controller
from src.plants.async_simulator import AsyncSimulator
from src.runner import Runner

bs, chz, t = 10, 100, 1
N = t * bs * chz
N_desc = t * chz


class BaseTest(unittest.TestCase):

    def async_plant_check(self, plant: AsyncSimulator, controller: Controller, x0: Sized):
        dof = len(x0)
        self._step_check(plant, x0)
        self._history_check(plant, controller, dof)

    def _step_check(self, plant, x0):
        # Step works
        x1 = plant.step(x0, 0, 0)
        self.assertEqual(x1.shape, (bs, len(x0)))

    def _history_check(self, plant, controller, dof):
        # Run simulation
        Runner(plant, controller).run()
        # Load history and prepare for plotting
        fields = plant.load_history()
        xs, xs_desc, ts, us = fields['xs'], fields['xs_desc'], fields['ts'], fields['us']
        self.assertEqual(ts.shape, (N_desc,))
        self.assertEqual(xs.shape, (N, dof))
        self.assertEqual(xs_desc.shape, (N_desc, dof))
        self.assertEqual(us.shape, (N_desc,))

    def async_discrete_plant_check(self, plant: AsyncSimulator, controller: Controller, x0: Sized):
        dof = len(x0)
        self._discrete_step_check(plant, x0)
        self._discrete_history_check(plant, controller, dof)

    def _discrete_step_check(self, plant, x0):
        # Step works
        x1 = plant.step(x0, 0, 0)
        self.assertEqual(x1.shape, (len(x0),))

    def _discrete_history_check(self, plant, controller, dof):
        # Run simulation
        Runner(plant, controller).run()
        # Load history and prepare for plotting
        fields = plant.load_history()
        xs, ts, us = fields['xs'], fields['ts'], fields['us']
        self.assertEqual(ts.shape, (N_desc,))
        self.assertEqual(xs.shape, (N_desc, dof))
        self.assertEqual(us.shape, (N_desc,))
