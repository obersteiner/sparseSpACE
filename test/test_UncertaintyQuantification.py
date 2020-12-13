import unittest
import numpy as np

import sparseSpACE
from sparseSpACE.Function import *
from sparseSpACE.spatiallyAdaptiveSingleDimension2 import *
from sparseSpACE.ErrorCalculator import *
from sparseSpACE.GridOperation import *


class TestUncertaintyQuantification(unittest.TestCase):
    def test_expectation_variance(self):
        # Let's select the three-dimensional discontinuous FunctionUQ
        # as problem function and let the input parameters be
        # normally distributed
        problem_function = FunctionUQ()
        dim = 3
        distributions = [("Normal", 0.2, 1.0) for _ in range(dim)]
        # a and b are the weighted integration domain boundaries.
        # They should be set according to the distribution.
        a = np.array([-np.inf] * dim)
        b = np.array([np.inf] * dim)

        # Create the grid operation and the weighted grid
        op = UncertaintyQuantification(problem_function, distributions, a, b)
        grid = GlobalTrapezoidalGridWeighted(a, b, op, boundary=False)
        # The grid initialization requires the weight functions from the
        # operation; since currently the adaptive refinement takes the grid from
        # the operation, it has to be passed here
        op.set_grid(grid)
        # Select the function for which the grid is refined;
        # here it is the expectation and variance calculation via the moments
        op.set_expectation_variance_Function()
        # Initialize the adaptive refinement instance and refine the grid until
        # it has at least 200 points
        combiinstance = SpatiallyAdaptiveSingleDimensions2(a, b, operation=op, norm=2, use_volume_weighting=True,
                                                           grid_surplusses=op.get_grid())
        lmax = 2
        error_operator = ErrorCalculatorSingleDimVolumeGuided()
        combiinstance.performSpatiallyAdaptiv(1, lmax,
            error_operator, tol=0, max_evaluations=200, print_output=False)

        # Calculate the expectation and variance with the adaptive sparse grid
        # weighted integral result
        (E,), (Var,) = op.calculate_expectation_and_variance(combiinstance)

        # Test if the results are similar to the reference values
        E_ref, Var_ref = (2.670603962589227, 8.813897872367328)
        assert abs(E - E_ref) < 0.3, E
        assert abs(Var - Var_ref) < 1.0, Var

    def test_pce(self):
        problem_function = FunctionUQ()
        dim = 3
        distributions = [("Normal", 0.2, 1.0) for _ in range(dim)]
        a = np.array([-np.inf] * dim)
        b = np.array([np.inf] * dim)
        op = UncertaintyQuantification(problem_function, distributions, a, b)
        grid = GlobalTrapezoidalGridWeighted(a, b, op, boundary=False)
        op.set_grid(grid)

        polynomial_degree_max = 2
        # The grid needs to be refined for the PCE coefficient calculation
        op.set_PCE_Function(polynomial_degree_max)

        combiinstance = SpatiallyAdaptiveSingleDimensions2(a, b, operation=op, norm=2,
                                                           grid_surplusses=op.get_grid())
        lmax = 2
        error_operator = ErrorCalculatorSingleDimVolumeGuided()
        combiinstance.performSpatiallyAdaptiv(1, lmax,
            error_operator, tol=0, max_evaluations=200, print_output=False)

        # Create the PCE approximation; it is saved internally in the operation
        op.calculate_PCE(None, combiinstance)
        # Calculate the expectation and variance with the PCE coefficients
        (E,), (Var,) = op.get_expectation_and_variance_PCE()

        # The PCE Variance differs from the actual variance
        E_ref, Var_ref = (2.66882233703942, 5.110498374118302)
        assert abs(E - E_ref) < 0.3, E
        assert abs(Var - Var_ref) < 1.0, Var


if __name__ == '__main__':
    unittest.main()
