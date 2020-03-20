from sys import path

path.append('../src/')
import numpy as np
from ErrorCalculator import *
from GridOperation import *
from StandardCombi import *
from sklearn import datasets
from SGppCompare import plot_comparison

# dimension of the problem
dim = 2

# define number of samples
size = 500

# define boundaries
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
# mean = np.array([0.0] * dim)
# sigma = np.array([0.25]*dim)
# cov = np.diag(sigma**2)
# data = np.random.multivariate_normal(mean, cov, size)

# scikit learn datasets
# data = datasets.make_moons(size, noise=0.1)
data = datasets.make_circles(size, noise=0.1)

# csv dataset file
# data = "Datasets/faithful.csv"
# SGpp values for dataset
# values = "Values/Circles_level_4_lambda_0.0.csv"

# define lambda
lambd = 0.01

# define level of combigrid
minimum_level = 1
maximum_level = 4

# define operation to be performed
operation = DensityEstimation(data, dim, lambd=lambd)

# create the combiObject and initialize it with the operation
combiObject = StandardCombi(a, b, operation=operation)

# perform the density estimation operation, has to be done before the printing and plotting
combiObject.perform_operation(minimum_level, maximum_level)

print("Plot of dataset:")
operation.plot_dataset()

print("Combination Scheme:")
# when you pass the operation the function also plots the contour plot of each component grid
combiObject.print_resulting_combi_scheme(operation=operation)

print("Sparse Grid:")
combiObject.print_resulting_sparsegrid(markersize=20)

print("Plot of density estimation")
# when contour = True, the contour plot is shown next to the 3D plot
combiObject.plot(contour=True)

print("Plot of comparison between sparseSpACE and SG++")
# plot comparison between sparseSpACE and SG++ result if path to SG++ values is given
# plot_comparison(dim=dim, data=data, values=values, combiObject=combiObject, plot_data=False, minimum_level=minimum_level, maximum_level=maximum_level, lambd=lambd, pointsPerDim=100)
