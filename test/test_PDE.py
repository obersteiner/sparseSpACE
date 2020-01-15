import unittest
from sys import path
path.append('./src/')
path.append('./PDE/')
from StandardCombi import *
from GridOperation import *
from PDE_Solver import Poisson

class testPDE(unittest.TestCase):

    def test_FEniCS(self):
        "Test solver by reproducing u = 1 + x^2 + 2y^2"

        tol = 1E-10
        poisson2D = Poisson('-6.0', 1, '1 + x[0]*x[0] + 2*x[1]*x[1]', 2, '1 + x[0]*x[0] + 2*x[1]*x[1]', 2)

        # Iterate over mesh sizes and degrees
        for grid in [[3, 3], [3, 5], [5, 3], [20, 20]]:
            for degree in 1, 2, 3:
                print("Solving on a %s mesh with P%d elements." %(grid, degree))
                # Compute solution
                poisson2D.solvePDE(grid, degree)
                # Compute maximum error at vertices
                error_max = poisson2D.computeMaxError()
                print("Max error: %d" %(error_max))
                # Check maximum error
                msg = 'error_max = %g' % error_max
                assert error_max < tol, msg

    def test_CombinationTechnique(self):
        raise NotImplementedError

if __name__ == "__main__":
    unittest.main()