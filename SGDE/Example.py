from sys import path

import numpy as np
from ErrorCalculator import *
from GridOperation import *
from sklearn import datasets

# dimension of the problem
dim = 3

# level of the grid
level = 4

# define integration domain boundaries
a = np.zeros(dim)
b = np.ones(dim)

# define equidistant grid
grid = TrapezoidalGrid(a=a, b=b, boundary=False)
grid.setCurrentArea(a, b, (2, 2,2))
points = grid.getPoints()
coord = grid.get_coordinates()
number = grid.levelToNumPoints((2, 2))
number2 = grid.levelToNumPoints((3, 1))
moons = datasets.make_moons(noise=0.05)

operation = DensityEstimation(grid, moons, dim, level)
operation.plot()
