import unittest

import numpy as np
from components.dataclass import Coord3d


class CoordTest(unittest.TestCase):
    def test_coordEq(self):
        a = Coord3d(1, 2, 3)
        b = Coord3d(1, 2, 3)
        self.assertEqual(a, b)

    def test_coordInit(self):
        a = Coord3d(1, 2, 3)
        b = Coord3d(np.array([1, 2, 3]))
        self.assertEqual(a, b)

    def test_unitVector(self):
        a = Coord3d(1, 2, 3)
        self.assertEqual(
            a.unitVector(), Coord3d(1 / (14 ** 0.5), 2 / (14 ** 0.5), 3 / (14 ** 0.5))
        )

    def test_angle_0(self):
        a = Coord3d(1, 0, 0)
        b = Coord3d(1, 0, 0)
        np.testing.assert_almost_equal(a.angleBtw(b), 0.0)

    def test_angle_45(self):
        a = Coord3d(1, 0, 0)
        b = Coord3d(1, 1, 0)
        np.testing.assert_almost_equal(a.angleBtw(b), 45.0)

    def test_angle_60(self):
        a = Coord3d(0, 0, 1)
        b = Coord3d(0, 3 ** 0.5, 1)
        np.testing.assert_almost_equal(a.angleBtw(b), 60.0)

    def test_angle_90(self):
        a = Coord3d(1, 0, 0)
        b = Coord3d(0, 1, 0)
        np.testing.assert_almost_equal(a.angleBtw(b), 90.0)

    def test_angle_180(self):
        a = Coord3d(1, 0, 0)
        b = Coord3d(-1, 0, 0)
        np.testing.assert_almost_equal(a.angleBtw(b), 180.0)
