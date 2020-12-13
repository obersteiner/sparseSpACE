import unittest
import sparseSpACE
from sparseSpACE.StandardCombi import *
import math
from sparseSpACE.Function import *

class TestStandardCombi(unittest.TestCase):
    def test_points(self):
        a = -3
        b = math.pi
        for d in range(2, 6):
            f = FunctionLinear([10 ** i for i in range(d)])
            operation = Integration(f, grid=TrapezoidalGrid(np.ones(d)*a, np.ones(d)*b, d), dim=d, reference_solution=f.getAnalyticSolutionIntegral(np.ones(d)*a, np.ones(d)*b))
            standardCombi = StandardCombi(np.ones(d)*a, np.ones(d)*b, print_output=False, operation=operation)
            for l in range(8 - d):
                for l2 in range(l+1):
                    #print(l,l2,d)
                    standardCombi.set_combi_parameters(l2, l)
                    standardCombi.check_combi_scheme()

    def test_integration(self):
        a = -3
        b = 7.3
        for d in range(2, 6):
            f = FunctionLinear([10 ** i for i in range(d)])
            operation = Integration(f, grid=TrapezoidalGrid(np.ones(d)*a, np.ones(d)*b, d), dim=d, reference_solution=f.getAnalyticSolutionIntegral(np.ones(d)*a, np.ones(d)*b))
            standardCombi = StandardCombi(np.ones(d)*a, np.ones(d)*b, print_output=False, operation=operation)
            for l in range(8 - d):
                for l2 in range(l+1):
                    scheme, error, integral  = standardCombi.perform_operation(l2, l)
                    rel_error = error/f.getAnalyticSolutionIntegral(np.ones(d)*a, np.ones(d)*b)
                    self.assertAlmostEqual(rel_error, 0.0, 13)

    def test_interpolation(self):
        a = -1
        b = 7
        for d in range(2, 5):
            f = FunctionLinear([10 * (i+1) for i in range(d)])
            operation = Integration(f, grid=TrapezoidalGrid(np.ones(d)*a, np.ones(d)*b), dim=d, reference_solution=f.getAnalyticSolutionIntegral(np.ones(d)*a, np.ones(d)*b))
            standardCombi = StandardCombi(np.ones(d)*a, np.ones(d)*b, print_output=False, operation=operation)
            for l in range(8 - d):
                for l2 in range(l+1):
                    standardCombi.set_combi_parameters(l2, l)
                    grid_coordinates = [np.linspace(a, b, 3, endpoint=False) for _ in range(d)]
                    interpolated_points = standardCombi.interpolate_grid(grid_coordinates)
                    grid_points = get_cross_product_list(grid_coordinates)
                    for component_grid in standardCombi.scheme:
                        interpolated_points_grid = standardCombi.interpolate_points(grid_points, component_grid)
                        for i, p in enumerate(grid_points):
                            factor = abs(f(p)[0] if f(p)[0] != 0 else 1)
                            self.assertAlmostEqual((f(p)[0] - interpolated_points_grid[i][0]) / factor, 0, 13)
                    for i, p in enumerate(grid_points):
                        factor = abs(f(p)[0] if f(p)[0] != 0 else 1)
                        self.assertAlmostEqual((f(p)[0] - interpolated_points[i][0])/factor, 0, 13)
                    interpolated_points = standardCombi(grid_points)
                    for i, p in enumerate(grid_points):
                        factor = abs(f(p)[0] if f(p)[0] != 0 else 1)
                        self.assertAlmostEqual((f(p)[0] - interpolated_points[i][0])/factor, 0, 13)

    def test_number_of_points(self):
        a = -3
        b = 7.3
        for d in range(2, 6):
            f = FunctionLinear([10 ** i for i in range(d)])
            operation = Integration(f, grid=TrapezoidalGrid(np.ones(d)*a, np.ones(d)*b, d), dim=d, reference_solution=f.getAnalyticSolutionIntegral(np.ones(d)*a, np.ones(d)*b))
            standardCombi = StandardCombi(np.ones(d)*a, np.ones(d)*b, print_output=False, operation=operation)
            for l in range(8 - d):
                for l2 in range(l+1):
                    standardCombi.set_combi_parameters(l2, l)
                    points, weights = standardCombi.get_points_and_weights()
                    self.assertEqual(len(points), standardCombi.get_total_num_points(distinct_function_evals=False))
                    self.assertEqual(len(points), len(weights))
                    for component_grid in standardCombi.scheme:
                        points, weights = standardCombi.get_points_and_weights_component_grid(component_grid.levelvector)
                        self.assertEqual(len(points), np.prod(standardCombi.grid.levelToNumPoints(component_grid.levelvector)))
                        self.assertEqual(standardCombi.get_num_points_component_grid(component_grid.levelvector, False), np.prod(standardCombi.grid.levelToNumPoints(component_grid.levelvector)))


if __name__ == '__main__':
    unittest.main()
