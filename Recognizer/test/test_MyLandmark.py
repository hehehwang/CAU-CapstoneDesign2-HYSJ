import unittest

import numpy as np
from components.dataclass import MyLandmark


class MyLandmarkTest(unittest.TestCase):
    def test_landmark_init(self):
        a = MyLandmark(1, 2, 0, 4)  # x, y, z, v
        b = MyLandmark(1, 2, 4)  # x, y, v
        self.assertEqual(str(a), str(b))

    def test_visualize_1(self):
        a = MyLandmark(1, 2, 3, 4)
        self.assertEqual(a.vis, 4)

    def test_visualize_2(self):
        a = MyLandmark(1, 2, 4)
        self.assertEqual(a.vis, 4)
