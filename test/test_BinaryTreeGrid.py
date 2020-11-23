import unittest
import sparseSpACE

from sparseSpACE.Extrapolation import GridBinaryTree


class TestBinaryTreeGrid(unittest.TestCase):
    def setUp(self) -> None:
        self.test_grids = [
            ([0.0, 0.25, 0.5, 1.0], [0.0, 0.25, 0.5, 0.75, 1.0]),
            ([0.0, 0.125, 0.25, 0.375, 0.5, 1.0], [0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0]),
            ([0.0, 0.25, 0.375, 0.5, 1.0], [0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0]),
            ([0.0, 0.125, 0.25, 0.375, 0.4375, 0.5, 1.0], [0.0, 0.125, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.75, 1.0])
        ]

        self.test_grid_levels = [
            ([0, 2, 1, 0], [0, 2, 1, 2, 0]),
            ([0, 3, 2, 3, 1, 0], [0, 3, 2, 3, 1, 2, 0]),
            ([0, 2, 3, 1, 0], [0, 3, 2, 3, 1, 2, 0]),
            ([0, 3, 2, 3, 4, 1, 0], [0, 3, 2, 4, 3, 4, 1, 2, 0])
        ]

        self.grid_binary_tree = GridBinaryTree()

    # This method is testing only grids that have not been rebalanced
    def test_tree_init(self):
        for (i, (grid, full_binary_tree_grid)) in enumerate(self.test_grids):
            grid_levels, full_grid_levels = self.test_grid_levels[i]
            self.grid_binary_tree.init_tree(grid, grid_levels)

            self.assertEqual(grid, self.grid_binary_tree.get_grid())
            self.assertEqual(grid_levels, self.grid_binary_tree.get_grid_levels())

    def test_full_tree_expansion(self):
        for (i, (grid, full_binary_tree_grid)) in enumerate(self.test_grids):
            grid_levels, full_grid_levels = self.test_grid_levels[i]
            self.grid_binary_tree.init_tree(grid, grid_levels)
            self.grid_binary_tree.force_full_tree_invariant()

            self.assertEqual(full_binary_tree_grid, self.grid_binary_tree.get_grid())
            self.assertEqual(full_grid_levels, self.grid_binary_tree.get_grid_levels())

    def test_tree_init_max_level(self):
        a = 0
        b = 1
        max_level = 3

        self.grid_binary_tree.init_perfect_tree_with_max_level(a, b, max_level)

        expected_grid = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
        expected_grid_levels = [0, 3, 2, 3, 1, 3, 2, 3, 0]

        self.assertEqual(expected_grid, self.grid_binary_tree.get_grid())
        self.assertEqual(expected_grid_levels, self.grid_binary_tree.get_grid_levels())

    def test_increment_level_in_each_subtree(self):
        grid = [0, 0.5, 0.75, 1]
        grid_levels = [0, 1, 2, 0]

        self.grid_binary_tree.init_tree(grid, grid_levels)
        self.grid_binary_tree.increment_level_in_each_subtree()

        expected_grid = [0, 0.5, 0.625, 0.75, 0.875, 1]
        expected_grid_levels = [0, 1, 3, 2, 3, 0]

        self.assertEqual(expected_grid, self.grid_binary_tree.get_grid())
        self.assertEqual(expected_grid_levels, self.grid_binary_tree.get_grid_levels())


if __name__ == '__main__':
    unittest.main()
