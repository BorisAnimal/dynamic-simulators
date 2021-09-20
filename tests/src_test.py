import unittest

import numpy as np

from src.trajectory import harmonic_trajectory_builder


class TestLift(unittest.TestCase):
    def test_controller(self):
        trajectory = harmonic_trajectory_builder(0, 1, 0.5 / np.pi)
        a, da, dda = trajectory(0)
        self.assertEqual(trajectory(0.0), (0.0, 1.0, 0.0))
        self.assertTrue(all(np.isclose(trajectory(np.pi / 2), (1.0, 0.0, -1.0))))
        self.assertTrue(all(np.isclose(trajectory(np.pi), (0.0, -1.0, 0.0))))
        self.assertTrue(all(np.isclose(trajectory(np.pi * 2), (0.0, 1.0, 0.0))))
