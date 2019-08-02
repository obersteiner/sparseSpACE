import unittest
from sys import path
path.append('../src/')
from combiScheme import *


class TestCombiScheme(unittest.TestCase):
    def test_size(self):
        combi_scheme = CombiScheme(dim=2)
        for l in range(10):
            for l2 in range(l+1):
                combi_grids = combi_scheme.getCombiScheme(lmin=l2, lmax=l, do_print=False)
                expected_num_grids = 2*(l - l2 + 1) - 1
                self.assertEqual(len(combi_grids), expected_num_grids)

    def test_coefficients(self):
        for d in range(2, 6):
            combi_scheme = CombiScheme(dim=d)
            for l in range(12 - d):
                for l2 in range(l+1):
                    combi_grids = combi_scheme.getCombiScheme(lmin=l2, lmax=l, do_print=False)
                    sum_of_coefficients = 0
                    for component_grid in combi_grids:
                        sum_of_coefficients += component_grid.coefficient
                    #print(l,l2)
                    self.assertEqual(sum_of_coefficients, 1)

    def test_size_adaptive(self):
        combi_scheme = CombiScheme(dim=2)
        for l in range(10):
            for l2 in range(l+1):
                combi_scheme.init_adaptive_combi_scheme(lmin=l2, lmax=l)
                combi_grids = combi_scheme.getCombiScheme(lmin=l2, lmax=l, do_print=False)
                expected_num_grids = 2*(l - l2 + 1) - 1
                sum_of_coefficients = 0
                for component_grid in combi_grids:
                    sum_of_coefficients += component_grid.coefficient
                self.assertEqual(len(combi_grids), expected_num_grids)

    def test_coefficients_adaptive(self):
        for d in range(2, 6):
            combi_scheme = CombiScheme(dim=d)
            for l in range(12 - d):
                for l2 in range(l+1):
                    combi_scheme.init_adaptive_combi_scheme(lmin=l2, lmax=l)
                    combi_grids = combi_scheme.getCombiScheme(lmin=l2, lmax=l, do_print=False)
                    sum_of_coefficients = 0
                    for component_grid in combi_grids:
                        sum_of_coefficients += component_grid.coefficient
                    self.assertEqual(sum_of_coefficients, 1)

    def test_adaptive_scheme_updates(self,):
        for d in range(2, 6):
            combi_scheme = CombiScheme(dim=d)
            for l in range(10 - d):
                for l2 in range(l+1):
                    combi_scheme.init_adaptive_combi_scheme(lmin=l2, lmax=l)
                    for i in range(10):
                        print(list(combi_scheme.active_index_set)[0])
                        combi_scheme.update_adaptive_combi(list(combi_scheme.active_index_set)[0])
                        combi_grids = combi_scheme.getCombiScheme(lmin=l2, lmax=l, do_print=False)
                        sum_of_coefficients = 0
                        for component_grid in combi_grids:
                            sum_of_coefficients += component_grid.coefficient
                        self.assertEqual(sum_of_coefficients, 1)

if __name__ == '__main__':
    unittest.main()