import unittest

import numpy as np
from components.landmarkHandler import MyLandmark


class MyLandmarkTest(unittest.TestCase):
    def test_visualize_1(self):
        a = MyLandmark(1, 2, 3, 4)
        self.assertEqual(a.vis, 4)

    def test_visualize_2(self):
        a = MyLandmark(1, 2, 4)
        self.assertEqual(a.y, 2)
