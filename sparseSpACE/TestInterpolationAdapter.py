import sparseSpACE
from sparseSpACE.Extrapolation import ExtrapolationGrid, GridVersion, SliceContainerVersion, SliceGrouping, SliceVersion
from sparseSpACE.Function import Polynomial1d

# TODO refactor in a test case


def run_extrapolation(grid, grid_levels, function, container_version: SliceContainerVersion):
    print("with {}".format(container_version))
    romberg_grid = ExtrapolationGrid(grid_version=GridVersion.INTERPOLATE_SUB_GRIDS,
                                     container_version=container_version,
                                     slice_grouping=SliceGrouping.GROUPED_OPTIMIZED,
                                     slice_version=SliceVersion.ROMBERG_DEFAULT,
                                     force_balanced_refinement_tree=True)
    romberg_grid.set_grid(grid, grid_levels)
    actual_result = romberg_grid.integrate(function)
    expected_result = 1.5

    # romberg_grid.plot_grid_with_containers()
    print("- expected result: {}".format(expected_result))
    print("- actual result: {}".format(actual_result))
    print("- delta: {}".format(actual_result - expected_result))
    print()


# --------------------------------------------------------
# Balanced adaptive grid
grid = [0, 0.5, 0.625, 0.75, 1]
grid_levels = [0, 1, 3, 2, 0]
function = Polynomial1d([1, 0, 0, 2])
print("Balanced adaptive grid")
print(grid)
print(grid_levels)
print()

# -- with Lagrange interpolation
# run_extrapolation(grid, grid_levels, function, SliceContainerVersion.LAGRANGE_ROMBERG)

# -- with hierarchical lagrange grid
run_extrapolation(grid, grid_levels, function, SliceContainerVersion.HIERARCHICAL_LAGRANGE_ROMBERG)

# -- with BSpline interpolation
run_extrapolation(grid, grid_levels, function, SliceContainerVersion.BSPLINE_ROMBERG)


# --------------------------------------------------------
# Unbalanced grid

# def test_exactness_on_unbalanced_adaptive_interpolated_lagrange_grid(self):
#     grid = [0, 0.5, 0.625, 0.75, 1]
#     grid_levels = [0, 1, 3, 2, 0]
#
#     function = Polynomial1d([1, 0, 0, 2])
#
#     romberg_grid = ExtrapolationGrid(grid_version=GridVersion.INTERPOLATE_SUB_GRIDS,
#                                      container_version=SliceContainerVersion.LAGRANGE_ROMBERG,
#                                      slice_grouping=SliceGrouping.GROUPED_OPTIMIZED,
#                                      slice_version=SliceVersion.ROMBERG_DEFAULT,
#                                      force_balanced_refinement_tree=False)
#     romberg_grid.set_grid(grid, grid_levels)
#     actual_result = romberg_grid.integrate(function)
#
#     expected_result = 1.5
#
#     # TODO works only with force_balanced_refinement_tree = True
#     # self.assertAlmostEqual(expected_result, actual_result)
#
#
# def test_exactness_on_unbalanced_adaptive_interpolated_bspline_grid(self):
#     grid = [0, 0.5, 0.625, 0.75, 1]
#     grid_levels = [0, 1, 3, 2, 0]
#
#     romberg_grid = ExtrapolationGrid(grid_version=GridVersion.INTERPOLATE_SUB_GRIDS,
#                                      container_version=SliceContainerVersion.BSPLINE_ROMBERG,
#                                      slice_grouping=SliceGrouping.GROUPED_OPTIMIZED,
#                                      slice_version=SliceVersion.ROMBERG_DEFAULT,
#                                      force_balanced_refinement_tree=False)
#     romberg_grid.set_grid(grid, grid_levels)
#     function = Polynomial1d([1, 0, 0, 2])
#
#     actual_result = romberg_grid.integrate(function)
#     expected_result = 1.5
#
#     # self.assertAlmostEqual(expected_result, actual_result)
#
#
# def test_exactness_on_unbalanced_adaptive_interpolated_hierarchical_lagrange_grid(self):
#     grid = [0, 0.5, 0.625, 0.75, 1]
#     grid_levels = [0, 1, 3, 2, 0]
#
#     romberg_grid = ExtrapolationGrid(grid_version=GridVersion.INTERPOLATE_SUB_GRIDS,
#                                      container_version=SliceContainerVersion.HIERARCHICAL_LAGRANGE_ROMBERG,
#                                      slice_grouping=SliceGrouping.GROUPED_OPTIMIZED,
#                                      slice_version=SliceVersion.ROMBERG_DEFAULT,
#                                      force_balanced_refinement_tree=True)
#     romberg_grid.set_grid(grid, grid_levels)
#     function = Polynomial1d([1, 0, 0, 2])
#
#     actual_result = romberg_grid.integrate(function)
#     expected_result = 1.5
#
#     # self.assertAlmostEqual(expected_result, actual_result)