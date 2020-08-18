from sys import path

path.append('src/')
from StandardCombi import *
import math
from Function import *
a=0
b=1
d=3
l2=1
l=3
f = FunctionLinear([10 ** i for i in range(d)])

operation = Integration(f, grid=TrapezoidalGrid(np.ones(d)*a, np.ones(d)*b, d), dim=d, reference_solution=f.getAnalyticSolutionIntegral(np.ones(d)*a, np.ones(d)*b))

standardCombi = StandardCombi(np.ones(d)*a, np.ones(d)*b, print_output=False, operation=operation)
standardCombi.set_combi_parameters(l2, l)
grid_coordinates = [np.linspace(a, b, 100, endpoint=False) for _ in range(d)]
grid_points = get_cross_product_list(grid_coordinates)
interpolated_points = standardCombi(grid_points)

