import unittest

from impl.utils.lift import lift, _rbf


class TestLift(unittest.TestCase):
    def test_call(self):
        lifted = lift([1, 2, 3], 100)
        self.assertEqual(lifted.shape, (103, 1))

    def test_rbf(self):
        rbf = _rbf([1, 2, 3], 100)
        self.assertEqual(rbf.shape, (100, 1))


if __name__ == '__main__':
    unittest.main()
