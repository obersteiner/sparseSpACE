from sys import path

import numpy as np
from ErrorCalculator import *
from GridOperation import *
from StandardCombi import *
from sklearn import datasets

# dimension of the problem
dim = 2

# define integration domain boundaries
a = np.zeros(dim)
b = np.ones(dim)

# define data
data = datasets.make_moons()

# define operation to be performed
operation = DensityEstimation(data, dim)

combiObject = StandardCombi(a, b, operation=operation)
minimum_level = 1
maximum_level = 4
combiObject.perform_operation(minimum_level, maximum_level)
print("Combination Scheme:")
combiObject.print_resulting_combi_scheme(markersize=5)
print("Sparse Grid:")
combiObject.print_resulting_sparsegrid(markersize=10)
print("Plot of combimodel for function:")
combiObject.plot()
