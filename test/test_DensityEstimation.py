import unittest
from sys import path

path.append('../src/')
from StandardCombi import *
from Grid import *
from GridOperation import DensityEstimation

dim = 2

# define integration domain boundaries
a = np.zeros(dim)
b = np.ones(dim)

# define data
data = datasets.make_moons()

# define operation to be performed
operation = DensityEstimation(data, dim)
operation.initialize()

combiObject = StandardCombi(a, b, operation=operation)
minimum_level = 1
maximum_level = 4

class TestDensityEstimation(unittest.TestCase):


    def test_r_matrix_size(self):
        return

if __name__ == '__main__':
    unittest.main()
