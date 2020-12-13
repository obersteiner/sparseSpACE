import math
import unittest
import sparseSpACE
from sparseSpACE.Extrapolation import BalancedExtrapolationGrid


class TestBalancedExtrapolationGrid(unittest.TestCase):
    def setUp(self):
        self.functions1d = [
            lambda x: 3 * x ** 5 + 4 * x + x ** 3 - 2,
            lambda x: math.exp(x ** 2) * x + 4 * x
        ]

    def test_refinement_tree_initialization(self):
        grid = [0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1]
        grid_levels = [0, 3, 2, 3, 1, 2, 0]

        extrapolation_grid = BalancedExtrapolationGrid()
        extrapolation_grid.set_grid(grid, grid_levels)

        self.assertEqual(grid, extrapolation_grid.get_grid())

    def test_weights_on_full_grid(self):
        grid = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
        grid_levels = [0, 3, 2, 3, 1, 3, 2, 3, 0]

        extrapolation_grid = BalancedExtrapolationGrid()
        extrapolation_grid.set_grid(grid, grid_levels)
        weights = extrapolation_grid.get_weights()

        # Compute integral value
        f = lambda x: 2 * x ** 3 + 1
        value = 0

        for i, point in enumerate(grid):
            value += weights[i] * f(point)

        self.assertAlmostEqual(1.5, value)

    # def test_weights_on_adaptive_balanced_grid(self):
    #     grid = [0.0, 0.125, 0.25, 0.375, 0.5,  0.75, 1]
    #     grid_levels = [0, 3, 2, 3, 1, 2, 0]
    #
    #     extrapolation_grid = BalancedExtrapolationGrid()
    #     extrapolation_grid.set_grid(grid, grid_levels)
    #     weights = extrapolation_grid.get_weights()
    #
    #     # Compute integral value
    #     f = lambda x: 2 * x ** 3 + 1
    #     value = 0
    #
    #     for i, point in enumerate(grid):
    #         value += weights[i] * f(point)
    #
    #     self.assertAlmostEqual(1.5, value)
    #
    # def test_weights_on_adaptive_grid(self):
    #     grid = [0, 0.5, 0.625, 0.75, 1]
    #     grid_levels = [0, 1, 3, 2, 0]
    #
    #     extrapolation_grid = BalancedExtrapolationGrid()
    #     extrapolation_grid.set_grid(grid, grid_levels)
    #     weights = extrapolation_grid.get_weights()
    #
    #     # Compute integral value
    #     f = lambda x: 2 * x ** 3 + 1
    #     value = 0
    #
    #     for i, point in enumerate(grid):
    #         value += weights[i] * f(point)
    #
    #     self.assertAlmostEqual(1.5, value)


if __name__ == '__main__':
    unittest.main()
