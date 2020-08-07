import unittest
from sys import path
path.append('../../src/')

from Extrapolation import SlicedRombergConstants, RombergGridSlice, ExtrapolationVersion
import numpy as np


class RombergConstants(unittest.TestCase):
    def setUp(self) -> None:
        self.places = 8

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Version: Romberg Sliced

    def test_constant_right_extrapolation(self):
        slice = RombergGridSlice([0, 0.5], [0, 1], [(0, 1), (0, 0.5)], ExtrapolationVersion.ROMBERG_SLICED)
        constants = SlicedRombergConstants(slice)

        # C_{1,1}
        support_points, constant_1_1 = constants.get_constant_1_for_right_boundary_extrapolation(1)
        self.assertEqual([0, 0.5], support_points)
        np.testing.assert_array_almost_equal(np.array([1/4, -1/4]), constant_1_1)

        # C_{2,1}
        support_points, constant_2_1 = constants.get_constant_2_for_right_boundary_extrapolation(1)
        self.assertEqual([0, 0.5], support_points)
        np.testing.assert_array_almost_equal(np.array([-1/16, 1/16]), constant_2_1)

        # C_{3,1}
        support_points, constant_3_1 = constants.get_constant_3_for_right_boundary_extrapolation(1)
        self.assertEqual([0, 0.5], support_points)
        np.testing.assert_array_almost_equal(np.array([1/4, -1/4]), constant_3_1)

        # C_{1,2}
        support_points, constant_1_2 = constants.get_constant_1_for_right_boundary_extrapolation(2)
        self.assertEqual([0, 0.5, 1], support_points)
        np.testing.assert_array_almost_equal(np.array([1/16, -1/8, 1/16]), constant_1_2)

        # C_{3, 2}
        support_points, constant_3_2 = constants.get_constant_3_for_right_boundary_extrapolation(2)
        self.assertEqual([0, 0.5, 1], support_points)
        np.testing.assert_array_almost_equal(np.array([-1/4, 1/2, -1/4]), constant_3_2)

        # C_{2}
        support_points, integration_constant = constants.get_integration_constant_weights(2)
        self.assertEqual([0, 0.5, 1], support_points)
        np.testing.assert_array_almost_equal(np.array([1/48, -1/24, 1/48]), integration_constant)


if __name__ == '__main__':
    unittest.main()
