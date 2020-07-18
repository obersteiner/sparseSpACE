from spatiallyAdaptiveSingleDimension2 import *
from GridOperation import *

# Settings
dim = 2
a = np.zeros(dim)
b = np.ones(dim)
max_tol = 10 ** (-20)  # set to max, for better comparison
max_evaluations = 10 ** 5
functions = []

# --- Smooth functions

# GenzCornerPeak: Coefficients from SGA Split Extend Paper, p. 18
coeffs = np.array([np.float64(3 * i) for i in range(1, dim + 1)])
f_genz_corner = GenzCornerPeak(coeffs=coeffs)

# GenzProductPeak: Coefficients from SGA Split Extend Paper, p. 18
coeffs = np.array([np.float64(3 * i) for i in range(1, dim + 1)])
midpoint = np.ones(dim) * 0.99
f_genz_product = GenzProductPeak(coeffs, midpoint)

# GenzContinious: Coefficients from SGA Split Extend Paper, p. 18
coeffs = np.array([np.float64(3 * i) for i in range(1, dim + 1)])
midpoint = np.ones(dim) * 0.5
f_genz_cont = GenzC0(coeffs, midpoint)

# GenzGaussian: Coefficients from SGA Split Extend Paper, p. 18
coeffs = np.array([np.float64(i) for i in range(1, dim + 1)])
midpoint = np.ones(dim) * 0.99
f_genz_gaussian = GenzGaussian(coeffs, midpoint)

# FunctionExpVar: See SGA Split Extend Paper, p. 18
f_exp_var = FunctionExpVar()

# GenzOszillatory: https://www.sfu.ca/~ssurjano/oscil.html
coeffs = np.array([np.float64(i) for i in range(1, dim + 1)])
offset = 0.5
f_genz_osz = GenzOszillatory(coeffs, offset)

# --- Discontinious functions

# GenzContinious: Coefficients from SGA Split Extend Paper, p. 18
border = np.ones(dim) * 0.2
coeffs = np.array([np.float64(3 * i) for i in range(1, dim + 1)])
f_genz_disc = GenzDiscontinious(border=border,coeffs=coeffs)

# Set function
f = f_genz_disc

# plot function
# f.plot(np.ones(dim)*a,np.ones(dim)*b)
reference_solution = f.getAnalyticSolutionIntegral(a, b)
errorOperator = ErrorCalculatorSingleDimVolumeGuided()

# Grids
grid=GlobalTrapezoidalGrid(a=a, b=b, modified_basis=False, boundary=True)
# grid = GlobalRombergGrid(a=a, b=b, modified_basis=False, boundary=True,
#                          slice_grouping=SliceGrouping.GROUPED,
#                          slice_version=SliceVersion.ROMBERG_DEFAULT,
#                          container_version=SliceContainerVersion.ROMBERG_DEFAULT)

operation = Integration(f=f, grid=grid, dim=dim, reference_solution=reference_solution)

# TODO Rebalancing conflicts with Romberg
adaptiveCombiInstanceSingleDim = SpatiallyAdaptiveSingleDimensions2(a, b,
                                                                    operation=operation, rebalancing=False,
                                                                    force_balanced_refinement_tree=False)

# performing the spatially adaptive refinement with the SingleDim method
adaptiveCombiInstanceSingleDim.performSpatiallyAdaptiv(1, 2, errorOperator,
                                                       max_tol,
                                                       max_evaluations=max_evaluations,
                                                       do_plot=True)

print("Number of points used in refinement:", adaptiveCombiInstanceSingleDim.get_total_num_points())