from sys import path
path.append('src/')
import numpy as np
from spatiallyAdaptiveSingleDimension2 import *
from Function import *
from ErrorCalculator import *
from GridOperation import *

# dimension of the problem
dim = 2

# define integration domain boundaries
a = np.zeros(dim)
b = np.ones(dim)

# define function to be integrated
midpoint = np.ones(dim) * 0.5
coefficients = np.array([ 10**0 * (d+1) for d in range(dim)])
f = GenzDiscontinious(border=midpoint,coeffs=coefficients)
# plot function
f.plot(np.ones(dim)*a,np.ones(dim)*b)

# reference integral solution for calculating errors
reference_solution = f.getAnalyticSolutionIntegral(a,b)

# define error estimator for refinement
errorOperator = ErrorCalculatorSingleDimVolumeGuided()

# define equidistant grid
grid=GlobalTrapezoidalGrid(a=a, b=b, modified_basis=False, boundary=True)

# NEW! define operation which shall be performed in the combination technique
operation = Integration(f=f, grid=grid, dim=dim, reference_solution=reference_solution)

# define SingleDim refinement strategy for Spatially Adaptive Combination Technique
adaptiveCombiInstanceSingleDim = SpatiallyAdaptiveSingleDimensions2(np.ones(dim) * a, np.ones(dim) * b, operation=operation)

# performing the spatially adaptive refinement with the SingleDim method
adaptiveCombiInstanceSingleDim.performSpatiallyAdaptiv(1,2,f,errorOperator,10**-2, do_plot=False)
