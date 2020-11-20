import unittest
import sparseSpACE
from sparseSpACE.combiScheme import *
from sparseSpACE.Utils import *

class TestUtils(unittest.TestCase):
    def test_size(self):
        for dim in range(1,6):
            for s in range(1,10):
                sizes = np.ones(dim, dtype=int) * s
                cross_product = get_cross_product_range_list(sizes)
                self.assertEqual(len(cross_product), s**dim)

                sizes = sizes + np.asarray(range(dim))
                cross_product = get_cross_product_range_list(sizes)
                self.assertEqual(len(cross_product), np.prod(sizes))

                arrays = [np.linspace(0, 1, sizes[d]) for d in range(dim)]
                cross_product = get_cross_product_list(arrays)
                self.assertEqual(len(cross_product), np.prod(sizes))

    def test_valid_entries(self):
        for dim in range(1, 6):
            for s in range(1, 10):
                sizes = np.ones(dim, dtype=int) * s + np.asarray(range(dim))
                cross_product = get_cross_product_range(sizes)
                for entry in cross_product:
                    for d in range(dim):
                        self.assertTrue(entry[d] in range(sizes[d]))

                arrays = [np.linspace(0, 1, sizes[d]) for d in range(dim)]
                cross_product = get_cross_product(arrays)
                for entry in cross_product:
                    for d in range(dim):
                        self.assertTrue(entry[d] in arrays[d])

    def test_combinations_only_occuring_once(self):
        for dim in range(1, 6):
            for s in range(1, 10):
                sizes = np.ones(dim, dtype=int) * s + np.asarray(range(dim))
                sets = [np.empty(sizes[d], dtype=set) for d in range(dim)]
                for d in range(dim):
                    for i in range(len(sets[d])):
                        sets[d][i] = set()
                cross_product = get_cross_product_range(sizes)
                for entry in cross_product:
                    for d in range(dim):
                        other_values = list(entry[:d]) + list(entry[d+1:])
                        self.assertTrue(tuple(other_values) not in sets[d][entry[d]])
                        sets[d][entry[d]].add(tuple(other_values))

                for d in range(dim):
                    for i in range(len(sets[d])):
                        sets[d][i] = set()
                arrays = [list(np.linspace(0, 1, sizes[d])) for d in range(dim)]
                cross_product = get_cross_product(arrays)
                for entry in cross_product:
                    for d in range(dim):
                        other_values = list(entry[:d]) + list(entry[d + 1:])
                        self.assertTrue(tuple(other_values) not in sets[d][arrays[d].index(entry[d])])
                        sets[d][arrays[d].index(entry[d])].add(tuple(other_values))


if __name__ == '__main__':
    unittest.main()