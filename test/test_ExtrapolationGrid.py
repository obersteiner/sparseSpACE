import math
import unittest
from scipy import integrate
import numpy as np

import sparseSpACE

from sparseSpACE.Function import Polynomial1d
from sparseSpACE.Extrapolation import ExtrapolationGrid, SliceVersion, SliceGrouping


# -----------------------------------------------------------------------------------------------------------------
# ---  Romberg Grid

class TestExtrapolationGrid(unittest.TestCase):
    def setUp(self):
        self.all_grids = [
            [0, 0.5, 0.625, 0.75, 1],  # Don't change, only add new grids. The tests below depend on this structure
        ]

        self.all_grid_levels = [
            [0, 1, 3, 2, 0],  # Don't change, only add new grid_levels. The tests below depend on this structure
        ]

    def get_grid_and_level(self, i):
        return self.all_grids[i], self.all_grid_levels[i]

    # This test case validates the selection of support points for the striped weight computation
    def test_support_sequence(self):
        grid, grid_levels = self.get_grid_and_level(0)

        romberg_grid = ExtrapolationGrid()
        romberg_grid.set_grid(grid, grid_levels)

        expected_sequences = [
            [(0, 1), (0, 1 / 2)],  # Slice 1
            [(0, 1), (1 / 2, 1), (1 / 2, 3 / 4), (1 / 2, 5 / 8)],  # Slice 2
            [(0, 1), (1 / 2, 1), (1 / 2, 3 / 4), (5 / 8, 3 / 4)],  # Slice 3
            [(0, 1), (1 / 2, 1), (3 / 4, 1)]  # Slice 4
        ]

        for i in range(len(expected_sequences)):
            actual_sequence = romberg_grid.compute_support_sequence(i, i + 1)
            self.assertEqual(expected_sequences[i], actual_sequence)

    def test_romberg_grid_weights(self):
        grid, grid_levels = self.get_grid_and_level(0)

        romberg_grid = ExtrapolationGrid(slice_grouping=SliceGrouping.UNIT,
                                         slice_version=SliceVersion.ROMBERG_DEFAULT)
        romberg_grid.set_grid(grid, grid_levels)

        actual_weights = romberg_grid.get_weights()
        expected_weights = [79 / 378, 194 / 567, 512 / 2835, 592 / 2835, 337 / 5670]

        np.testing.assert_almost_equal(expected_weights, actual_weights)

    def test_container_adjustment_with_correct_containers(self):
        grid = [0, 0.5, 0.625, 0.75, 1]
        grid_levels = [0, 1, 3, 2, 0]

        romberg_grid = ExtrapolationGrid(slice_grouping=SliceGrouping.GROUPED,
                                         slice_version=SliceVersion.ROMBERG_DEFAULT)
        romberg_grid.set_grid(grid, grid_levels)

        self.assertEqual(1, len(romberg_grid.slice_containers[0].slices))
        self.assertEqual(2, len(romberg_grid.slice_containers[1].slices))
        self.assertEqual(1, len(romberg_grid.slice_containers[2].slices))

    def test_container_adjustment_with_incorrect_containers(self):
        # Grid with 3 slices in the first container
        grid = [0.0, 0.0625, 0.125, 0.25, 0.375, 0.5, 0.75, 0.875, 1]
        grid_levels = [0, 4, 3, 2, 3, 1, 2, 3, 0]

        romberg_grid = ExtrapolationGrid(slice_grouping=SliceGrouping.GROUPED_OPTIMIZED,
                                         slice_version=SliceVersion.ROMBERG_DEFAULT,
                                         force_balanced_refinement_tree=False)
        romberg_grid.set_grid(grid, grid_levels)

        # Containers:
        # [(0.0, 0.0625), (0.0625, 0.125)], [(0.125, 0.25), (0.25, 0.375)],
        # [(0.375, 0.5)], [(0.5, 0.75)], [(0.75, 0.875), (0.875, 1)]
        self.assertEqual(2, len(romberg_grid.slice_containers[0].slices), "Wrong slice count in first container")
        self.assertEqual(2, len(romberg_grid.slice_containers[1].slices), "Wrong slice count in 2nd container")
        self.assertEqual(1, len(romberg_grid.slice_containers[2].slices), "Wrong slice count in 3d container")
        self.assertEqual(1, len(romberg_grid.slice_containers[3].slices), "Wrong slice count in 4th container")
        self.assertEqual(2, len(romberg_grid.slice_containers[4].slices), "Wrong slice count in 5th container")

    def test_adjacent_containers(self):
        grid = [0.0, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 0.875, 1]
        grid_levels = [0, 3, 4, 2, 3, 1, 2, 3, 0]
        # Containers: [0, 0.125], [0.125, .0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1]

        romberg_grid = ExtrapolationGrid(slice_grouping=SliceGrouping.GROUPED_OPTIMIZED,
                                         slice_version=SliceVersion.ROMBERG_DEFAULT,
                                         force_balanced_refinement_tree=False)
        romberg_grid.set_grid(grid, grid_levels)

        containers = romberg_grid.slice_containers

        self.assertEqual([], containers[0].adjacent_containers_left)
        self.assertEqual([
            containers[1], containers[2], containers[3], containers[4]
        ], containers[0].adjacent_containers_right)

        self.assertEqual([
            containers[2], containers[1], containers[0]
        ], containers[3].adjacent_containers_left)
        self.assertEqual([
            containers[4]
        ], containers[3].adjacent_containers_right)

    def test_adjacent_slices(self):
        grid = [0.0, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 0.875, 1]
        grid_levels = [0, 3, 4, 2, 3, 1, 2, 3, 0]
        # Containers: [0, 0.125], [0.125, 0.1875, 0.25], [0.25, 0.375, 0.5], [0.5, 0.75], [0.75, 0.875, 1]

        romberg_grid = ExtrapolationGrid(slice_grouping=SliceGrouping.GROUPED_OPTIMIZED,
                                         slice_version=SliceVersion.ROMBERG_DEFAULT,
                                         force_balanced_refinement_tree=False)
        romberg_grid.set_grid(grid, grid_levels)

        containers = romberg_grid.slice_containers

        # Container 0
        self.assertEqual(None, containers[0].slices[0].adjacent_slice_left)
        self.assertEqual(containers[1].slices[0], containers[0].slices[0].adjacent_slice_right)

        # Container 1
        self.assertEqual(containers[0].slices[0], containers[1].slices[0].adjacent_slice_left)
        self.assertEqual(containers[1].slices[1], containers[1].slices[0].adjacent_slice_right)

        self.assertEqual(containers[1].slices[0], containers[1].slices[1].adjacent_slice_left)
        self.assertEqual(containers[2].slices[0], containers[1].slices[1].adjacent_slice_right)

        # Container 2
        self.assertEqual(containers[1].slices[-1], containers[2].slices[0].adjacent_slice_left)
        self.assertEqual(containers[2].slices[1], containers[2].slices[0].adjacent_slice_right)

        self.assertEqual(containers[2].slices[0], containers[2].slices[1].adjacent_slice_left)
        self.assertEqual(containers[3].slices[0], containers[2].slices[1].adjacent_slice_right)

        # Container 3
        self.assertEqual(containers[2].slices[-1], containers[3].slices[0].adjacent_slice_left)
        self.assertEqual(containers[4].slices[0], containers[3].slices[0].adjacent_slice_right)

        # Container 4
        self.assertEqual(containers[3].slices[-1], containers[4].slices[0].adjacent_slice_left)
        self.assertEqual(containers[4].slices[1], containers[4].slices[0].adjacent_slice_right)

        self.assertEqual(containers[4].slices[0], containers[4].slices[1].adjacent_slice_left)
        self.assertEqual(None, containers[4].slices[1].adjacent_slice_right)

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Approximation

    def test_integral_approximation(self):
        grid, grid_levels = self.get_grid_and_level(0)

        function = Polynomial1d([1, 0, 0, 2])

        romberg_grid = ExtrapolationGrid(slice_grouping=SliceGrouping.UNIT,
                                         slice_version=SliceVersion.ROMBERG_DEFAULT)
        romberg_grid.set_grid(grid, grid_levels)
        actual_result = romberg_grid.integrate(function)

        expected_value = 1388 / 945  # computed by hand. This is not the analytical result!!
        self.assertAlmostEqual(expected_value, actual_result)

    def test_integral_approximation_with_grouped_slices(self):
        grid, grid_levels = self.get_grid_and_level(0)

        function = Polynomial1d([1, 0, 0, 2])

        romberg_grid = ExtrapolationGrid(slice_grouping=SliceGrouping.GROUPED)
        romberg_grid.set_grid(grid, grid_levels)
        actual_result = romberg_grid.integrate(function)

        expected_value = 11279 / 7680  # computed by hand. This is not the analytical result!!
        self.assertEqual(expected_value, actual_result)

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Exactness

    def test_exactness_on_full_grid(self):
        grid = [1, 1.5, 2, 2.5, 3]
        grid_levels = [0, 2, 1, 2, 0]

        function = Polynomial1d([1, 0, 0, 2])  # Polynomial1d of degree 3

        for slice_grouping in SliceGrouping:
            romberg_grid = ExtrapolationGrid(slice_grouping=slice_grouping,
                                             slice_version=SliceVersion.ROMBERG_DEFAULT)
            romberg_grid.set_grid(grid, grid_levels)
            romberg_grid.integrate(function)

            self.assertEqual(0, romberg_grid.get_absolute_error())  # Romberg is exact for this grid

    def test_exactness_on_full_binary_tree_grid_with_grouped_slices(self):
        grid = [1, 2, 2.25, 2.5, 3]
        grid_levels = [0, 1, 3, 2, 0]

        function = Polynomial1d([1, 0, 0, 2])  # Polynomial1d of degree 3

        romberg_grid = ExtrapolationGrid(slice_grouping=SliceGrouping.GROUPED,
                                         force_balanced_refinement_tree=True)
        romberg_grid.set_grid(grid, grid_levels)
        romberg_grid.integrate(function)

        self.assertAlmostEqual(0, romberg_grid.get_absolute_error())  # Romberg is exact for this grid

    def test_exactness_on_adaptive_grid_with_grouped_slices(self):
        grid, grid_levels = self.get_grid_and_level(0)

        function = Polynomial1d([1, 0, 0, 2])

        romberg_grid = ExtrapolationGrid(slice_grouping=SliceGrouping.GROUPED)
        romberg_grid.set_grid(grid, grid_levels)
        romberg_grid.integrate(function)

        actual_error_in_container = romberg_grid.slice_containers[1].get_extrapolated_error()

        self.assertAlmostEqual(0, actual_error_in_container)

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Helpers for Romberg Grid

    def test_set_grid(self):
        grid, grid_levels = self.get_grid_and_level(0)

        romberg_grid = ExtrapolationGrid()
        romberg_grid.set_grid(grid, grid_levels)

        self.assertEqual(grid, romberg_grid.get_grid())
        self.assertEqual(grid_levels, romberg_grid.get_grid_levels())
        self.assertEqual(grid[0], romberg_grid.a)
        self.assertEqual(grid[-1], romberg_grid.b)

    def test_set_function(self):
        grid, grid_levels = self.get_grid_and_level(0)
        function1 = Polynomial1d([1, 0, 0, 2])
        function2 = Polynomial1d([3, 0, 1, 2])

        romberg_grid = ExtrapolationGrid()
        romberg_grid.set_function(function1)
        romberg_grid.set_grid(grid, grid_levels)
        containers = romberg_grid.slice_containers

        for container in containers:
            self.assertEqual(function1, container.function)

        romberg_grid.integrate(function2)
        for container in containers:
            self.assertEqual(function2, container.function)

        romberg_grid.set_function(function1)
        for container in containers:
            self.assertEqual(function1, container.function)


# -----------------------------------------------------------------------------------------------------------------
# ---  Unit Test

if __name__ == '__main__':
    unittest.main()
