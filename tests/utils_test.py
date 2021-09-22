import unittest

import numpy as np

from impl.utils.lift import lift, _rbf
from impl.utils.rk import rk


class TestLift(unittest.TestCase):
    def test_lift_call(self):
        lifted = lift(np.zeros((3, 1)), 100)
        self.assertEqual(lifted.shape, (103, 1))

    def test_rbf_call(self):
        rbf = _rbf(np.zeros((3, 2)), 100)
        self.assertEqual(rbf.shape, (100, 2))

    def test_rbf_stability(self):
        x = np.random.rand(3, 2)
        rbf1 = _rbf(x, 100)
        rbf2 = _rbf(x, 100)
        self.assertTrue((rbf1 == rbf2).all())

    def test_rk(self):
        df = lambda x, t, u: np.array([100, 100])
        res = rk(df, 0, np.array([0, 0]), u=None, dt=0.01)
        self.assertTrue((res == np.array([1, 1])).all())


if __name__ == '__main__':
    unittest.main()
