import unittest
import numpy as np
import chaospy as cp

import sparseSpACE
from sparseSpACE.Function import *
from sparseSpACE.spatiallyAdaptiveSingleDimension2 import *
from sparseSpACE.ErrorCalculator import *
from sparseSpACE.GridOperation import *


class TestIntegrationUQ(unittest.TestCase):
    def test_normal_integration(self):
        #print("Calculating an expectation with an Integration Operation")
        d = 2
        bigvalue = 7.0
        a = np.array([-bigvalue, -bigvalue])
        b = np.array([bigvalue, bigvalue])

        distr = []
        for _ in range(d):
            distr.append(cp.Normal(0,2))
        distr_joint = cp.J(*distr)
        f = FunctionMultilinear([2.0, 0.0])
        fw = FunctionCustom(lambda coords: f(coords)[0]
            * float(distr_joint.pdf(coords)))

        grid = GlobalBSplineGrid(a, b)
        op = Integration(fw, grid=grid, dim=d)

        error_operator = ErrorCalculatorSingleDimVolumeGuided()
        combiinstance = SpatiallyAdaptiveSingleDimensions2(a, b, operation=op)
        #print("performSpatiallyAdaptiv…")
        v = combiinstance.performSpatiallyAdaptiv(1, 2, error_operator, tol=10**-3,
            max_evaluations=40, min_evaluations=25, do_plot=False, print_output=False)
        integral = v[3][0]
        #print("expectation", integral)


if __name__ == '__main__':
    unittest.main()
