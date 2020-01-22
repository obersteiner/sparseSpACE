import unittest
from sys import path
path.append('./src/')
path.append('./PDE/')
from StandardCombi import *
from GridOperation import PDE_Solve
from ComponentGridInfo import *
from Grid import GlobalGrid
from PDE_Solver import Poisson
import itertools as it
import numpy as np

class testPDE(unittest.TestCase):

    def test_FEniCS(self):
        "Test vanilla FEniCS solver by reproducing analytic solution u_exact = 1 + x^2 + 2y^2"
        tol = 1E-10
        poisson2D = Poisson('-6.0', 1, '1 + x[0]*x[0] + 2*x[1]*x[1]', 2, '1 + x[0]*x[0] + 2*x[1]*x[1]', 2)

        # Iterate over mesh sizes and degrees
        for grid in [[3, 3], [3, 5], [5, 3], [20, 20]]:
            for degree in 1, 2, 3:
                print("Solving on a %s mesh with P%d elements." %(grid, degree))
                # Compute solution
                poisson2D.solve(grid, degree)
                # Compute maximum error at vertices
                error_max = poisson2D.computeMaxError()
                print("Max error: %d" %(error_max))
                # Check maximum error
                msg = 'error_max = %g' % error_max
                assert error_max < tol, msg

    def test_CombinationTechnique(self):
        ''' Test Combination technique '''
        dim = 2
        a = np.zeros(dim)
        b = np.ones(dim)
        grid = GlobalGrid(a=a, b=b, boundary=True) # As for now grid is just a dummy nedded to initialize operation

        for maxlv, minlv in it.combinations([1,2,3,4],2):
            n = (2**maxlv)*np.ones(dim, dtype=int)
            poisson2D = Poisson('-6.0', 1, '1 + x[0]*x[0] + 2*x[1]*x[1]', 2, '1 + x[0]*x[0] + 2*x[1]*x[1]', 2)
            # Reference solution
            poisson2D.solve(n)
            reference_solution = poisson2D.get_vertex_values()
            # Combi solution
            operation = PDE_Solve(solver=poisson2D, maxlv=tuple(maxlv*np.ones(dim, dtype=int)), grid=grid, reference_solution=reference_solution)
            combiObject = StandardCombi(a, b, operation=operation)
            # Solve PDE using standard Combination Technique
            combiObject.perform_operation(minlv, maxlv)
            # print("Max error: %d" %(error_max))


if __name__ == "__main__":
    unittest.main()