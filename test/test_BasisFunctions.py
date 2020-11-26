import unittest
import sparseSpACE
from sparseSpACE.BasisFunctions import *

class TestBasisFunctions(unittest.TestCase):
    def test_lagrange_basis(self):
        for n in range(10):
            for i in range(n):
                points = np.linspace(0, 1 , n)
                basis = LagrangeBasis(p=n - 1, index=i, knots=points)
                for j in range(n):
                    # check basic property of lagrange polynomials p_i(x_j) = \delta(i,j)
                    if i == j:
                        self.assertAlmostEqual(basis(points[j]), 1.0)
                    else:
                        self.assertEqual(basis(points[j]), 0.0)

    def test_lagrange_basis_restricted(self):
        for n in range(10):
            for i in range(n):
                points = np.linspace(0, 1 , n)
                basis = LagrangeBasisRestricted(p=n - 1, index=i, knots=points)
                for j in range(n):
                    # check basic property of lagrange polynomials p_i(x_j) = \delta(i,j)
                    if i == j:
                        self.assertAlmostEqual(basis(points[j]), 1.0)
                    else:
                        self.assertEqual(basis(points[j]), 0.0)
                # restricted polynomials p_i are 0 when x < x_{i-1} or x > x_{i+1}
                points2 = np.linspace(0, 1 , 10 * n)
                for j in range(len(points2)):
                    if points2[j] < points[max(i - 1, 0)] or points2[j] > points[min(i+1, len(points) - 1)]:
                        self.assertEqual(basis(points2[j]), 0.0)

    def test_lagrange_basis_restricted_modified(self):
        for n in range(1, 10):
            for i in range(1, n+1):
                points = np.linspace(0, 1 , n + 2)
                basis = LagrangeBasisRestrictedModified(p=n + 1, index=i, knots=points, a=0, b=1, level=n)
                if i == 1:  # test extrapolation to left
                    self.assertTrue(basis(points[0]) > 0)
                if i == n:  # test extrapolation to right
                    self.assertTrue(basis(points[n+1]) > 0)
                for j in range(1, n+1):
                    # check basic property of lagrange polynomials p_i(x_j) = \delta(i,j)
                    if i == j:
                        self.assertAlmostEqual(basis(points[j]), 1.0)
                    else:
                        self.assertEqual(basis(points[j]), 0.0)
                # restricted polynomials p_i are 0 when x < x_{i-1} or x > x_{i+1}
                points2 = np.linspace(0, 1 , 10 * n)
                for j in range(len(points2)):
                    if points2[j] < points[max(i - 1, 0)] or points2[j] > points[min(i+1, len(points) - 1)]:
                        self.assertEqual(basis(points2[j]), 0.0)


if __name__ == '__main__':
    unittest.main()