from spatiallyAdaptiveSingleDimension2 import *
from GridOperation import *

# Spatially adaptive single dimension
# dimension of the problem
dim = 2

# define integration domain boundaries
a = np.zeros(dim)
b = np.ones(dim)

# Smooth functions
# coeffs = np.ones(dim)
# midpoint = np.ones(dim) * 0.5
# f = GenzProductPeak(coeffs, midpoint)

# coeffs = np.ones(dim)
# f = GenzCornerPeak(coeffs=coeffs)

# coeffs = np.ones(dim)
# offset = 1
# f = GenzOszillatory(coeffs, offset)

# coeffs = np.ones(dim)
# midpoint = np.ones(dim) * 0.5
# f = GenzGaussian(coeffs, midpoint)

# Discontinuous functions
midpoint = np.ones(dim) * 0.5
coefficients = np.array([10**0 * (d+1) for d in range(dim)])
f = GenzDiscontinious(border=midpoint, coeffs=coefficients)

# plot function
# f.plot(np.ones(dim)*a,np.ones(dim)*b)
reference_solution = f.getAnalyticSolutionIntegral(a, b)
errorOperator = ErrorCalculatorSingleDimVolumeGuided()

# Grids
# grid=GlobalTrapezoidalGrid(a=a, b=b, modified_basis=False, boundary=True)
grid = GlobalRombergGrid(a=a, b=b, modified_basis=False, boundary=True,
                         slice_grouping=SliceGrouping.GROUPED,
                         slice_version=SliceVersion.TRAPEZOID,
                         container_version=SliceContainerVersion.ROMBERG_DEFAULT)

operation = Integration(f=f, grid=grid, dim=dim, reference_solution=reference_solution)

# TODO Rebalancing conflicts with Romberg
adaptiveCombiInstanceSingleDim = SpatiallyAdaptiveSingleDimensions2(np.ones(dim) * a, np.ones(dim) * b,
                                                                    operation=operation, rebalancing=False,
                                                                    force_full_binary_tree_grid=False)

# performing the spatially adaptive refinement with the SingleDim method
adaptiveCombiInstanceSingleDim.performSpatiallyAdaptiv(1, 2, errorOperator,
                                                       10**-3, do_plot=True)

print("Number of points used in refinement:", adaptiveCombiInstanceSingleDim.get_total_num_points())