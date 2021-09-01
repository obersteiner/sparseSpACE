import unittest
import numpy as np
import sparseSpACE
from sparseSpACE.StandardCombi import *
from sparseSpACE.Grid import *
from sparseSpACE.GridOperation import DensityEstimation
from sparseSpACE.GridOperation import Regression
from sparseSpACE.ErrorCalculator import ErrorCalculatorSingleDimMisclassificationGlobal, ErrorCalculatorSingleDimVolumeGuided
from sparseSpACE.spatiallyAdaptiveSingleDimension2 import SpatiallyAdaptiveSingleDimensions2

import numpy as np
from matplotlib import pyplot as plt


class TestRegression(unittest.TestCase):

    def test_calculate_alphas(self):
        """
        Test the calculated surpluses of the component grids for a combination scheme of level 4
        """
        data = np.array([[0.25, 0.25], [0.25, 0.5], [0.25, 0.75],
                         [0.5, 0.25], [0.5, 0.5], [0.5, 0.75],
                         [0.75, 0.25], [0.75, 0.5], [0.75, 0.75]])

        target = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.])

        regression = Regression(data=data, target_values=target, regularization=0, regularization_matrix='C')
        regression.training_target_values = target
        regression.training_data = data

        numPoints = 2 ** (np.asarray([2, 2], dtype=int))
        numPoints -= 1
        regression.grid.numPoints = numPoints

        alphas = regression.solve_regression([2, 2])

        assert len(target) == len(alphas)
        for i in range(len(target)):
            target[i] == alphas[i]

    def test_calculate_alphas_smooth(self):
        """
        Test the calculated surpluses of the component grids for a combination scheme of level 4
        """
        data = np.array([[0.25, 0.25], [0.25, 0.5], [0.25, 0.75],
                         [0.5, 0.25], [0.5, 0.5], [0.5, 0.75],
                         [0.75, 0.25], [0.75, 0.5], [0.75, 0.75]])

        target = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.])

        regression = Regression(data=data, target_values=target, regularization=0.1, regularization_matrix='C')
        regression.training_target_values = target
        regression.training_data = data

        numPoints = 2 ** (np.asarray([2, 2], dtype=int))
        numPoints -= 1
        regression.grid.numPoints = numPoints

        alphas = regression.solve_regression_smooth([2, 2])

        assert len(target) == len(alphas)
        for i in range(len(target)):
            target[i] == alphas[i]


    def test_calculate_C_matrix(self):
        """
        Test the C matrix entries
        """

        regression = Regression(data=np.array([[0.3]]), target_values=np.array([1]), regularization=0.1, regularization_matrix='C')

        levelvec = [1,2]

        numPoints = 2 ** (np.asarray(levelvec, dtype=int))
        numPoints -= 1
        regression.grid.numPoints = numPoints

        C = regression.build_C_matrix(levelvec)

        # completely overlapping
        self.assertAlmostEqual(C[0][0], C[1][1])
        self.assertAlmostEqual(C[2][2], C[1][1])
        self.assertAlmostEqual(C[0][0], 2.66666667)
        # no overlap
        self.assertEqual(C[0][2], 0.)
        self.assertEqual(C[2][0], 0.)
        # partly overlapping
        self.assertEqual(C[1][0], C[2][1])
        self.assertEqual(C[0][1], C[1][2])
        self.assertEqual(C[1][0], -0.3333333333333333)

    def test_Opticom_sum_always_1(self):
        """
        Test that the sum of the coefficients (component grids) is always 1 after Opticom
        """

        regression = Regression(data=np.array([[0.3], [0.3], [0.3], [0.3]]), target_values=np.array([1, 1, 1, 1]), regularization=0.1, regularization_matrix='C')

        combiObject = regression.train(0.5, 1, 3, False)


        # Opticom Garcke
        regression.optimize_coefficients(combiObject, 1)
        sum = 0.
        for i in range(len(combiObject.scheme)):
            sum += combiObject.scheme[i].coefficient
        self.assertAlmostEqual(1., sum)

        # Opticom least squares without regularization
        regression.optimize_coefficients(combiObject, 2)
        sum = 0.
        for i in range(len(combiObject.scheme)):
            sum += combiObject.scheme[i].coefficient
        self.assertAlmostEqual(1., sum)

        # Opticom error based
        regression.optimize_coefficients(combiObject, 3)
        sum = 0.
        for i in range(len(combiObject.scheme)):
            sum += combiObject.scheme[i].coefficient
        self.assertAlmostEqual(1., sum)

    def test_Opticom_sum_always_1_spatially_adaptive(self):
        """
        Test that the sum of the coefficients (component grids) is always 1 after Opticom
        """

        regression = Regression(data=np.array([[0.3], [0.3], [0.3], [0.3]]), target_values=np.array([1, 1, 1, 1]), regularization=0.1, regularization_matrix='C')

        combiObject = regression.train_spatially_adaptive(0.5, 0.5,10**-5, 0, False, False)


        # Opticom Garcke
        regression.optimize_coefficients_spatially_adaptive(combiObject, 1)
        sum = 0.
        for i in range(len(combiObject.scheme)):
            sum += combiObject.scheme[i].coefficient
        self.assertAlmostEqual(1., sum)

        # Opticom least squares without regularization
        regression.optimize_coefficients_spatially_adaptive(combiObject, 2)
        sum = 0.
        for i in range(len(combiObject.scheme)):
            sum += combiObject.scheme[i].coefficient
        self.assertAlmostEqual(1., sum)

        # Opticom error based
        regression.optimize_coefficients_spatially_adaptive(combiObject, 3)
        sum = 0.
        for i in range(len(combiObject.scheme)):
            sum += combiObject.scheme[i].coefficient
        self.assertAlmostEqual(1., sum)

    def test_calculate_C_matrix_spatially_adaptive(self):
        """
        Test the C matrix entries
        """

        regression = Regression(data=np.array([[0.3]]), target_values=np.array([1]), regularization=0.1, regularization_matrix='C')

        C = regression.build_C_matrix_dimension_wise([[0., 0.25, 0.5, 0.75, 1.]], [])

        # fully overlapping
        self.assertAlmostEqual(C[0][0], C[1][1])
        self.assertAlmostEqual(C[0][0], C[2][2])
        self.assertAlmostEqual(C[0][0], 8.)

        # partly overlapping
        self.assertAlmostEqual(C[0][1], C[1][0])
        self.assertAlmostEqual(C[1][0], C[2][1])
        self.assertAlmostEqual(C[1][2], C[2][1])
        self.assertAlmostEqual(C[0][1], -4.)

        # not overlapping
        self.assertAlmostEqual(C[0][2], C[2][0])
        self.assertAlmostEqual(C[0][2], -2.)

    def test_regularization_term_garcke(self):
        """
        Test the regularization term of Opticom Garcke
        """

        regression = Regression(data=np.array([[0.3,0.3]]), target_values=np.array([1]), regularization=0.1, regularization_matrix='C')

        levelvec = [1,2]

        numPoints = 2 ** (np.asarray(levelvec, dtype=int))
        numPoints -= 1
        regression.grid.numPoints = numPoints

        value = regression.sum_C_matrix_with_alphas(levelvec, [1, 1, 1], [1, 1, 1])
        self.assertAlmostEqual(value, 6.666666666666666)


        value = regression.sum_C_matrix_with_alphas(levelvec, [2, 1, 1], [1, 1, 2])
        self.assertAlmostEqual(value, 21.33333333333333)


    def test_regularization_term_garcke_spatially_adaptive(self):
        """
        Test the C matrix entries
        """

        regression = Regression(data=np.array([[0.3]]), target_values=np.array([1]), regularization=0.1, regularization_matrix='C')

        value = regression.sum_C_matrix_with_alphas_spatially_adaptive([[0., 0.25, 0.5, 0.75, 1.]], [1, 1, 1], [1, 1, 1])
        self.assertAlmostEqual(value, 4.)


        value = regression.sum_C_matrix_with_alphas_spatially_adaptive([[0., 0.25, 0.5, 0.75, 1.]], [2, 2, 1], [1, 1, 2])
        self.assertAlmostEqual(value, -16.)



if __name__ == '__main__':
    unittest.main()
