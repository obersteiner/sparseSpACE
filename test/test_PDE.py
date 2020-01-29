import unittest
from sys import path
path.append('./src/')
path.append('./PDE/')
from StandardCombi import *
from GridOperation import PDE_Solve
from ComponentGridInfo import *
from Grid import TrapezoidalGrid
from PDE_Solver import Poisson, GaussianHill
import itertools as it
import numpy as np

class testPDE(unittest.TestCase):

    def test_Poisson_FEniCS(self):
        "Test vanilla FEniCS solver by reproducing analytic solution u_exact = 1 + x^2 + 2y^2"
        print("Testing FEniCS Poisson solver")
        tol = 1E-10
        poisson2D = Poisson('-6.0', 1, '1 + x[0]*x[0] + 2*x[1]*x[1]', 2, '1 + x[0]*x[0] + 2*x[1]*x[1]', 2)

        # Iterate over mesh sizes and degrees
        for grid in [[3, 3], [3, 5], [5, 3], [20, 20]]:
            for degree in 1, 2, 3:
                print("Solving on a %s mesh with P%d elements." %(grid, degree))
                # Compute solution
                poisson2D.define_unit_hypercube_mesh(N=grid, degree = degree)
                poisson2D.solve()
                # Compute maximum error at vertices
                error_max = poisson2D.computeMaxError()
                print("Max error: %d" %(error_max))
                # Check maximum error
                msg = 'error_max = %g' % error_max
                assert error_max < tol, msg

    def test_Poisson_CombinationTechnique(self):
        ''' Test Combination technique '''
        print("Testing combination technique with Poisson solver")
        dim = 2
        a = np.zeros(dim)
        b = 2*np.ones(dim)
        grid = TrapezoidalGrid(a=a, b=b, boundary=True)

        poisson2D = Poisson('-6.0', 1, '1 + x[0]*x[0] + 2*x[1]*x[1]', 2, '1 + x[0]*x[0] + 2*x[1]*x[1]', 2)

        for minlv, maxlv in [(1,3),(1,4),(1,5)]:
            print("Solving for minlv,maxlv = (%d,%d)" %(minlv, maxlv))
            grid.setCurrentArea(a, b, maxlv*np.ones(dim, dtype=int))      
            #reference solution on a full grid
            poisson2D.define_rectangle_mesh(a,b, *(len(n) for n in grid.get_coordinates()))
            reference_solution = poisson2D.solve()
            # Combi solution
            operation = PDE_Solve(solver=poisson2D, grid=grid, reference_solution=reference_solution)
            combiObject = StandardCombi(a, b, operation=operation, print_output=False)
            _,_, combi_result = combiObject.perform_operation(minlv, maxlv)
            error = operation.get_error(combi_result, reference_solution)
            print("Max error: %d" %(error))

    # def test_GaussianHill_FEniCS(self):
    #     "Test vanilla FEniCS solver by reproducing analytic solution u_exact = 1 + x^2 + 2y^2"
    #     tol = 1E-10
    #     gauss = GauussianHill()
        

if __name__ == "__main__":
    unittest.main()