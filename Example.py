from sys import path

path.append('src/')
import numpy as np
from spatiallyAdaptiveSingleDimension2 import *
from Function import *
from ErrorCalculator import *
from GridOperation import *
from sklearn import datasets

# dimension of the problem
dim = 2

# define integration domain boundaries
a = np.zeros(dim)
b = np.ones(dim)

# define function to be integrated
midpoint = np.ones(dim) * 0.5
coefficients = np.array([10 ** 0 * (d + 1) for d in range(dim)])
f = GenzDiscontinious(border=midpoint, coeffs=coefficients)
# plot function
# f.plot(np.ones(dim) * a, np.ones(dim) * b)

# reference integral solution for calculating errors
reference_solution = f.getAnalyticSolutionIntegral(a, b)

# define error estimator for refinement
errorOperator = ErrorCalculatorSingleDimVolumeGuided()

# define equidistant grid
grid = TrapezoidalGrid(a=a, b=b, boundary=False)
grid.setCurrentArea(a, b, (2, 2))
points = grid.getPoints()
coord = grid.get_coordinates()
number = grid.levelToNumPoints((2, 2))
number2 = grid.levelToNumPoints((3, 1))
moons = datasets.make_moons(noise=0.05)

# NEW! define operation which shall be performed in the combination technique
operation = DensityEstimation(grid, moons, 2,4)
operation.plot()

# define SingleDim refinement strategy for Spatially Adaptive Combination Technique
adaptiveCombiInstanceSingleDim = SpatiallyAdaptiveSingleDimensions2(np.ones(dim) * a, np.ones(dim) * b,
                                                                    operation=operation)

# performing the spatially adaptive refinement with the SingleDim method
adaptiveCombiInstanceSingleDim.performSpatiallyAdaptiv(1, 2, f, errorOperator, 10 ** -2, do_plot=False)
