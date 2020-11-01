import unittest
from sys import path

path.append('../src/')
from StandardCombi import *
from Grid import *
from GridOperation import DensityEstimation

import numpy as np
from matplotlib import pyplot as plt


class TestDensityEstimation(unittest.TestCase):

    def test_adjacency(self):
        """
        Test the adjacency for the hat function at index (2,2) (middle) with other indices
        for the grid for the level vector (2,2) (below)
        +-------------------+
        |  O      O      O  |
        |                   |
        |  O      O      O  |
        |                   |
        |  O      O      O  |
        +-------------------+
        """
        dim = 2

        # define integration domain boundaries
        a = np.zeros(dim)
        b = np.ones(dim)

        lvec = (2, 2)
        index = (2, 2)
        operation = DensityEstimation("", dim)
        operation.grid.setCurrentArea(a, b, lvec)
        for i in operation.grid.get_indexlist():
            self.assertTrue(operation.check_adjacency(index, i))
        self.assertFalse(operation.check_adjacency(index, (0, 1)))
        self.assertFalse(operation.check_adjacency(index, (4, 1)))
        self.assertFalse(operation.check_adjacency(index, (1, 4)))
        self.assertFalse(operation.check_adjacency(index, (1, 0)))
        self.assertFalse(operation.check_adjacency(index, (4, 4)))

    def test_get_hats(self):
        """
        Test the returned hat functions for different points
        for the grid for the level vector (2,2) (below)
        +-------------------+
        |  O      O      O  |
        |                   |
        |  O      O      O  |
        |                   |
        |  O      O      O  |
        +-------------------+
        """
        dim = 2

        # define integration domain boundaries
        a = np.zeros(dim)
        b = np.ones(dim)

        lvec = (2, 2)
        operation = DensityEstimation("", dim)
        operation.grid.setCurrentArea(a, b, lvec)
        self.assertCountEqual(operation.get_hats_in_support(lvec,  np.array([0.3, 0.3])), [(1, 1), (2, 1), (2, 2), (1, 2)])
        self.assertCountEqual(operation.get_hats_in_support(lvec,  np.array([0.1, 0.24])), [(1, 1)])
        self.assertCountEqual(operation.get_hats_in_support(lvec,  np.array([0.6, 0.8])), [(2, 3), (3, 3)])
        self.assertCountEqual(operation.get_hats_in_support(lvec,  np.array([0.01, 0.51])), [(1, 3), (1, 2)])
        self.assertCountEqual(operation.get_hats_in_support(lvec,  np.array([0.999, 0.999])), [(3, 3)])
        self.assertFalse(operation.get_hats_in_support(lvec, np.array([1.01, 1.01])))
        self.assertFalse(operation.get_hats_in_support(lvec,  np.array([-0.01, -0.01])))

    def test_DE(self):
        """
        Test the calculated surpluses of the component grids for a combination scheme of level 4
        """
        dim = 2

        # define integration domain boundaries
        a = np.zeros(dim)
        b = np.ones(dim)

        # define data
        data = datasets.make_moons()

        # define operation to be performed
        operation = DensityEstimation(data, dim)
        operation.initialize()

        levelvecs = [(1, 4), (2, 3), (3, 2), (4, 1), (1, 3), (2, 2), (3, 1)]
        # New values with build_R_matrix explicit calculation
        alphas = {(1, 4): [3.31741396, 1.16132214, 1.13819061, 0.79472702, 0.8495253,
                           1.37687227, 0.76602162, 1.29277621, 0.76602162, 1.37687227,
                           0.8495253, 0.79472702, 1.13819061, 1.16132214, 3.31741396],
                  (2, 3): [-0.48892477, 0.55555412, 0.62763947, 0.74510083, 1.13562281,
                           -0.15835321, 1.9659157, 2.82292207, 0.58184326, 0.9169748,
                           0.60578953, 0.9169748, 0.58184326, 2.82292207, 1.9659157,
                           -0.15835321, 1.13562281, 0.74510083, 0.62763947, 0.55555412,
                           -0.48892477],
                  (3, 2): [0.29445604, -0.32813262, 3.46769424, -0.80084934, 0.00708973,
                           -0.1158526, 2.49797987, 3.29111459, -0.04394134, 1.84550343,
                           -2.81996655, 1.84550343, -0.04394134, 3.29111459, 2.49797987,
                           -0.1158526, 0.00708973, -0.80084934, 3.46769424, -0.32813262,
                           0.29445604],
                  (4, 1): [2.41196947, 0.15360354, 0.38484969, -0.38270344, 1.5540324,
                           3.34256214, 0.26867418, 0.78548498, 0.26867418, 3.34256214,
                           1.5540324, -0.38270344, 0.38484969, 0.15360354, 2.41196947],
                  (1, 3): [2.3181979, 0.51260238, 1.2343326, 0.93744408, 1.2343326,
                           0.51260238, 2.3181979],
                  (2, 2): [0.22791796, 0.87156074, 0.8927848, 1.73623453, 0.2709627,
                           1.73623453, 0.8927848, 0.87156074, 0.22791796],
                  (3, 1): [1.20572347, -0.33933214, 2.97721107, -0.37835152, 2.97721107,
                           -0.33933214, 1.20572347]}

        for i in range(len(levelvecs)):
            operation.grid.setCurrentArea(np.zeros(len(levelvecs[i])), np.ones(len(levelvecs[i])), levelvecs[i])

            R = operation.build_R_matrix(levelvecs[i])
            b = operation.calculate_B(operation.data, levelvecs[i])

            numbOfPoints = np.prod(operation.grid.levelToNumPoints(levelvecs[i]))

            # Check matrix size
            self.assertEqual(numbOfPoints, len(R))
            self.assertEqual(numbOfPoints, len(R[0]))
            self.assertEqual(numbOfPoints * numbOfPoints, np.prod(R.shape))
            self.assertEqual(numbOfPoints, len(b))

            alpha1 = solve(R, b)
            alpha2 = alphas.get(levelvecs[i])

            self.assertEqual(len(alpha1), len(alpha2))

            # Check values
            for j in range(len(alpha1)):
                self.assertAlmostEqual(alpha1[j], alpha2[j])

    def test_calculate_L2_scalarproduct(self):

        DE = DensityEstimation(data=[], dim=1)
        dom_1 = [(-1.0, 1.0)]
        point_1 = [0.0]
        res = DE.calculate_L2_scalarproduct(point_i=point_1, domain_i=dom_1,
                                            point_j=point_1, domain_j=dom_1)
        self.assertAlmostEqual((2.0 / 3.0) - res[0], 0.0)

        DE = DensityEstimation(data=[], dim=2)
        dom_1 = [(-1.0, 1.0), (-1.0, 1.0)]
        point_1 = [0.0, 0.0]
        res = DE.calculate_L2_scalarproduct(point_i=point_1, domain_i=dom_1,
                                            point_j=point_1, domain_j=dom_1)
        self.assertAlmostEqual((4.0 / 9.0) - res[0], 0.0)

if __name__ == '__main__':
    unittest.main()
