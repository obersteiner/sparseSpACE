import unittest

import sparseSpACE
from sparseSpACE.Extrapolation import SliceGrouping, ExtrapolationGrid, SliceVersion, SliceContainerVersion
from sparseSpACE.Function import Polynomial1d

# -----------------------------------------------------------------------------------------------------------------
# ---  Simpson Romberg Grid

class TestExtrapolationGrid(unittest.TestCase):
    # -----------------------------------------------------------------------------------------------------------------
    # ---  Exactness

    def test_exactness_on_full_grid(self):
        grid = [1, 1.5, 2, 2.5, 3]
        grid_levels = [0, 2, 1, 2, 0]

        function = Polynomial1d([1, 0, 0, 2])  # Polynomial1d of degree 3

        for slice_grouping in SliceGrouping:
            romberg_grid = ExtrapolationGrid(slice_grouping=slice_grouping,
                                             slice_version=SliceVersion.ROMBERG_DEFAULT,
                                             container_version=SliceContainerVersion.SIMPSON_ROMBERG)
            romberg_grid.set_grid(grid, grid_levels)
            romberg_grid.integrate(function)

            self.assertAlmostEqual(0, romberg_grid.get_absolute_error(), 1)


# -----------------------------------------------------------------------------------------------------------------
# ---  Unit Test

if __name__ == '__main__':
    unittest.main()
