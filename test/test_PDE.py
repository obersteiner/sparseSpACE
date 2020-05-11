import unittest
from sys import path
path.append('./src/')
path.append('./PDE/')
from StandardCombi import *
from GridOperation import PDE_Solve
from ComponentGridInfo import *
from Grid import TrapezoidalGrid
from PDE_Solver import *
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
                poisson2D.define_unit_hypercube_mesh(N=grid)
                poisson2D.solve(el_deg=degree)

                # Compute maximum error at vertices
                error_max = poisson2D.computeMaxError()
                print("Max error: %d" %(error_max))

                # Check maximum error
                msg = 'Tolerance exceeded: error_max = %g' % error_max
                assert error_max < tol, msg


    def test_Poisson_CombinationTechnique(self):
        ''' Test Combination technique with Poisson solver'''
        print("Testing combination technique with Poisson solver")
        tol = 1E-10
        dim = 2
        a=np.zeros(dim)
        b=np.ones(dim)
        grid = TrapezoidalGrid(a=a, b=b, boundary=True)
        poisson2D = Poisson('-6.0', 1, '1 + x[0]*x[0] + 2*x[1]*x[1]', 2, '1 + x[0]*x[0] + 2*x[1]*x[1]', 2)

        for minlv, maxlv in [(1,3),(1,4),(1,5),(2,4),(2,5)]:
            print("Solving for minlv,maxlv = {}".format((minlv, maxlv)))
            grid.setCurrentArea(a, b, maxlv*np.ones(dim, dtype=int))     

            # Reference solution on a full grid
            poisson2D.define_rectangle_mesh(a,b, *(len(n) for n in grid.get_coordinates()))
            reference_solution = poisson2D.solve()

            # Combi solution
            operation = PDE_Solve(solver=poisson2D, dim=dim, grid=grid, reference_solution=reference_solution)
            combiObject = StandardCombi(a, b, operation=operation, print_output=False)
            _,_, combi_result = combiObject.perform_operation(minlv, maxlv)

            # Error
            error = operation.compute_difference(combi_result, reference_solution,2)
            print("Max error: {0:.10f}".format(error))

            # Check maximum error
            msg = 'Tolerance exceeded: error_max = %g' % error
            assert error < tol, msg


    def test_GaussianHill(self):
        ''' Test Combination technique with GaussianHill solver'''
        print("Testing combination technique with Gaussian_Hill solver")
        tol = 1E-10
        dim = 2
        a = -1*np.ones(dim)
        b = np.ones(dim)
        grid = TrapezoidalGrid(a=a, b=b, boundary=True)
        tol = 0.01
        gauss = Gaussian_Hill()

        for minlv, maxlv in [(1,3),(1,4),(1,5),(2,4),(2,5)]:
            print("Solving for minlv,maxlv = {}".format((minlv, maxlv)))
            grid.setCurrentArea(a, b, maxlv*np.ones(dim, dtype=int)) 

            # Reference solution
            gauss.define_rectangle_mesh(a,b, *(len(n) for n in grid.get_coordinates()))
            reference_solution = gauss.solve()

            # Combi setup and solution
            operation = PDE_Solve(solver=gauss, grid=grid, dim=dim, reference_solution=reference_solution)
            combiObject = StandardCombi(a, b, operation=operation, print_output=False)
            __,__, combi_result = combiObject.perform_operation(minlv, maxlv)

            #Error
            error = operation.compute_difference(combi_result, reference_solution,2)
            print("Max error: %d" %(error))

            # Check maximum error
            if error > tol:
                print('Tolerance exceeded: error_max = %g' % error)

    def test_NavierStokes(self):
        ''' Test Combination technique with NavierStokes solver'''
        print("Testing combination technique with NavierStokes solver")
        tol = 1E-10
        dim = 2
        a = np.zeros(dim)
        b = np.ones(dim)
        grid = TrapezoidalGrid(a=a, b=b, boundary=True)
        tol = 0.01
        navierStokes = Navier_Stokes(t_max=5)

        for minlv, maxlv in [(1,3),(1,4),(1,5),(2,4),(2,5)]:
            print("Solving for minlv,maxlv = {}".format((minlv, maxlv)))
            grid.setCurrentArea(a, b, maxlv*np.ones(dim, dtype=int)) 

            # Reference solution
            navierStokes.define_rectangle_mesh(a,b, *(len(n) for n in grid.get_coordinates()))
            reference_solution = navierStokes.solve()

            # Combi setup and solution
            operation = PDE_Solve(solver=navierStokes, grid=grid, dim=dim, reference_solution=reference_solution)
            combiObject = StandardCombi(a, b, operation=operation, print_output=False)
            __,__, combi_result = combiObject.perform_operation(minlv, maxlv)

            #Error
            error = operation.compute_difference(combi_result, reference_solution,2)
            print("Max error: %d" %(error))

            # Check maximum error
            if error > tol:
                print('Tolerance exceeded: error_max = %g' % error)


if __name__ == "__main__":
    unittest.main()