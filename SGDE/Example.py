from sys import path

path.append('../src/')
import numpy as np
from ErrorCalculator import *
from GridOperation import *
from StandardCombi import *
from sklearn import datasets

# dimension of the problem
dim = 2

# define number of samples
size = 500

# define integration domain boundaries
a = np.zeros(dim)
b = np.ones(dim)

# define data (https://docs.scipy.org/doc/numpy-1.14.0/reference/routines.random.html)
# random floats
# data = np.random.random((size, dim))

# samples from the standard exponential distribution.
# data = np.random.standard_exponential((size, dim))

# samples from the standard exponential distribution
# data = np.random.standard_normal((size, dim))

# multivariate normal distribution
mean = [0, 0]
cov = [[1, 0], [0, 100]]
data = np.random.multivariate_normal(mean, cov, size)

# scikit learn datasets
# data = datasets.make_moons(size)
# data = datasets.make_circles(size)

# define operation to be performed
operation = DensityEstimation(data, dim)

# create the combiObject and initialize it with the operation
combiObject = StandardCombi(a, b, operation=operation)

# define level of combigrid
minimum_level = 1
maximum_level = 4

# perform the density estimation operation
combiObject.perform_operation(minimum_level, maximum_level)
print("Combination Scheme:")
combiObject.print_resulting_combi_scheme()
print("Sparse Grid:")
combiObject.print_resulting_sparsegrid()
print("Plot of dataset:")
operation.plot_dataset()
print("Plot of density estimation")
combiObject.plot()
