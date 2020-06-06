import math
from collections import defaultdict
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class Function(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_analytical_solution(self, a, b):
        pass

    @abstractmethod
    def evaluate_at(self, point):
        pass

    @abstractmethod
    def evaluate_antiderivative_at(self, point):
        pass


# -----------------------------------------------------------------------------------------------------------------
# ---  Extrapolation Coefficients

class ExtrapolationVersion(Enum):
    ROMBERG_DEFAULT = 1
    ROMBERG_LINEAR = 2
    ROMBERG_SLICED = 3


class ExtrapolationCoefficients(ABC):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    @abstractmethod
    def get_coefficient(self, m, j):
        pass

    def get_step_width(self, j):
        return (self.b - self.a) / (2 ** j)


class RombergDefaultCoefficients(ExtrapolationCoefficients):
    def __init__(self, a, b):
        super(RombergDefaultCoefficients, self).__init__(a, b)

    def get_coefficient(self, m, j):
        coefficient = 1
        h_j = self.get_step_width(j)

        for i in range(m + 1):
            h_i = self.get_step_width(i)
            coefficient *= (h_i ** 2) / (h_i ** 2 - h_j ** 2) if i != j else 1

        return coefficient


class RombergLinearCoefficients(ExtrapolationCoefficients):
    def __init__(self, a, b):
        super(RombergLinearCoefficients, self).__init__(a, b)

    def get_coefficient(self, m, j):
        coefficient = 1
        h_j = self.get_step_width(j)

        for i in range(m + 1):
            h_i = self.get_step_width(i)
            coefficient *= (h_i ** 1) / (h_i ** 1 - h_j ** 1) if i != j else 1

        return coefficient


class RombergSlicedCoefficients(ExtrapolationCoefficients):
    def __init__(self, a, b, support_sequence=None):
        super(RombergSlicedCoefficients, self).__init__(a, b)

        self.support_sequence = support_sequence

    # TODO Consider constant remainder terms in error expansion
    def get_coefficient(self, max_level, j):
        coefficient = 1
        slice_boundary_left, slice_boundary_right = self.support_sequence[-1]

        midpoint = (slice_boundary_left + slice_boundary_right) / 2
        assert max_level == len(self.support_sequence) - 1

        left_j, right_j = self.support_sequence[j]
        H_j = right_j - left_j

        for i in range(max_level + 1):
            left_i, right_i = self.support_sequence[i]

            if i > 0:
                left_prev, right_prev = self.support_sequence[i - 1]
            else:
                left_prev = None

            H_i = right_i - left_i

            if left_i == left_prev and left_prev is None:
                # Coefficient for refinement of b_i (right support)
                coefficient *= H_j / (H_j - H_i) if i != j else 1
            else:
                d_j = H_j / (left_j - midpoint) ** 2
                d_i = H_i / (left_i - midpoint) ** 2

                # Coefficient for refinement of a_i (left support)
                coefficient *= d_j / (d_j - d_i) if i != j else 1

        return coefficient


class ExtrapolationCoefficientsFactory:
    def __init__(self, extrapolation_version: ExtrapolationVersion):
        self.extrapolation_version = extrapolation_version

    def get(self, a, b, support_sequence=None) -> ExtrapolationCoefficients:
        if self.extrapolation_version == ExtrapolationVersion.ROMBERG_DEFAULT:
            return RombergDefaultCoefficients(a, b)

        elif self.extrapolation_version == ExtrapolationVersion.ROMBERG_LINEAR:
            return RombergLinearCoefficients(a, b)

        elif self.extrapolation_version == ExtrapolationVersion.ROMBERG_SLICED:
            assert support_sequence is not None, "Please provide support point sequence for slice extrapolation."
            return RombergSlicedCoefficients(a, b, support_sequence)

        else:
            raise RuntimeError("Wrong ExtrapolationVersion provided.")


class RombergWeightFactory:
    def __init__(self, a, b, version: ExtrapolationVersion = ExtrapolationVersion.ROMBERG_DEFAULT):
        self.a = a
        self.b = b
        self.version = version
        self.extrapolation_factory = ExtrapolationCoefficientsFactory(version).get(a, b)

    def get_boundary_point_weight(self, max_level):
        weight = 0

        for j in range(max_level + 1):
            coefficient = self.extrapolation_factory.get_coefficient(max_level, j)
            step_width = self.get_step_width(j)
            weight += (coefficient * step_width) / 2

        return weight

    def get_inner_point_weight(self, level, max_level):
        assert level >= 1

        weight = 0

        for j in range(level, max_level + 1):
            coefficient = self.extrapolation_factory.get_coefficient(max_level, j)
            step_width = self.get_step_width(j)
            weight += coefficient * step_width

        return weight

    def get_extrapolation_coefficient(self, m, j):
        return self.extrapolation_factory.get_coefficient(m, j)

    def get_step_width(self, j):
        return self.extrapolation_factory.get_step_width(j)


# -----------------------------------------------------------------------------------------------------------------
# ---  Binary Tree Grid


class GridBinaryTree:
    class __GridBinaryTree:
        # TODO test caching
        def __init__(self, use_caching=False, print_debug=False):
            self.grid = None
            self.grid_levels = None
            self.a = None
            self.b = None

            # Root node has level 1, since boundary points are note store in the tree
            self.root_node = None
            self.active_node_queue = []

            # Stores mapping of grid and its (full) tree grid
            self.tree_dict = {}

            self.use_caching = use_caching
            self.print_debug = print_debug

        def __init_attributes(self, grid, grid_levels):
            self.grid = grid
            self.grid_levels = grid_levels
            self.a = grid[0]
            self.b = grid[-1]

            self.root_node = None
            self.active_node_queue = []
            self.full_tree_dict = {}

        # Given a grid level array, this method initializes a appropriate binary tree structure
        def init_tree(self, grid, grid_levels):
            self.__init_attributes(grid, grid_levels)

            # Remove boundary levels
            assert grid_levels[0] == 0 and grid_levels[-1] == 0

            # build tree based on inner points (neglect index 0 and -1)
            self.__init_tree_rec(None, grid, grid_levels, 1, len(grid_levels) - 2)

            assert self.root_node is not None

        def __init_tree_rec(self, node, grid, grid_levels,
                            start_index, stop_index,
                            inside_left_subtree=None):
            # Find index of minimal element in active interval
            current_grid_level_slice = grid_levels[start_index:stop_index + 1]
            if start_index < stop_index:
                split_index = start_index + current_grid_level_slice.index(min(current_grid_level_slice))
            else:
                split_index = start_index

            # Termination criterion
            if stop_index < start_index or split_index > stop_index or split_index < start_index:
                return
            # Root node
            elif node is None:
                self.root_node = GridBinaryTree.__GridBinaryTree.Node(grid[split_index], print_debug=self.print_debug)
                self.__init_tree_rec(self.root_node, grid, grid_levels,
                                     start_index, split_index - 1, inside_left_subtree=True)
                self.__init_tree_rec(self.root_node, grid, grid_levels,
                                     split_index + 1, stop_index, inside_left_subtree=False)
            # Other nodes
            else:
                if inside_left_subtree:
                    node.set_left_child(GridBinaryTree.__GridBinaryTree.Node(grid[split_index], parent=node))
                    node = node.get_left_child()
                else:
                    node.set_right_child(GridBinaryTree.__GridBinaryTree.Node(grid[split_index], parent=node))
                    node = node.get_right_child()

                self.__init_tree_rec(node, grid, grid_levels,
                                     start_index, split_index - 1, inside_left_subtree=True)
                self.__init_tree_rec(node, grid, grid_levels,
                                     split_index + 1, stop_index, inside_left_subtree=False)

        # Expand tree until it is a full binary tree
        def force_full_tree_invariant(self):
            if self.root_node is None:
                return

            if self.use_caching and self.__is_cached(self.grid):
                return

            nodes = self.root_node.get_nodes_using_dfs_in_order()

            for node in nodes:
                if node.has_only_one_child():
                    if node.has_left_child():
                        point = node.point + (node.point - node.left_child.point)
                        right_child = GridBinaryTree.__GridBinaryTree.Node(point)
                        node.set_right_child(right_child)
                    elif node.has_right_child():
                        point = node.point - (node.right_child.point - node.point)
                        left_child = GridBinaryTree.__GridBinaryTree.Node(point)
                        node.set_left_child(left_child)
                    else:
                        raise Exception("This shouldn't happen in a binary tree")

            if self.use_caching:
                self.__cache_grid()

        # Method for numerical experiments and testing
        # This method initialises a perfect binary tee (each node has two children, all leafs are on max_level)
        def init_perfect_tree_with_max_level(self, a, b, max_level):
            queue = []

            self.a = a
            self.b = b

            root_point = (b - a) / 2
            self.root_node = GridBinaryTree.__GridBinaryTree.Node(root_point)
            queue.append(self.root_node)

            # Boundary points are already created before
            for i in range(1, max_level):
                queue = self.__increment_level_in_each_subtree(queue)

            assert self.root_node is not None

        # This method creates two new children for each leaf node
        #   => level of each subtree is increased by one
        # Note: This works for each tree type (default, full, perfect)
        # Important: non-leaf nodes (e.g. nodes with one children) are not considered!!!
        def increment_level_in_each_subtree(self):
            leafs = self.__get_leafs()

            self.__increment_level_in_each_subtree(leafs)

        def __increment_level_in_each_subtree(self, node_queue=[]):
            new_queue = []
            H = self.b - self.a

            while len(node_queue) > 0:
                node = node_queue.pop(0)

                node.set_left_child(GridBinaryTree.__GridBinaryTree.Node(
                    node.point - H / (2 ** (node.level + 1))
                ))

                node.set_right_child(GridBinaryTree.__GridBinaryTree.Node(
                    node.point + H / (2 ** (node.level + 1))
                ))

                new_queue.append(node.get_left_child())
                new_queue.append(node.get_right_child())

            return new_queue

        # Return grid with boundary points
        def get_grid(self):
            if not self.use_caching:
                return self.__get_grid()

            dict_key = self.__get_dict_key(self.grid)

            if self.__is_cached(self.grid):
                return self.__get_grid_from_cache(dict_key)

            self.__cache_grid()

            return self.__get_grid_from_cache(dict_key)

        def __get_grid(self):
            return [self.a] + self.root_node.get_grid() + [self.b]

        # Return levels, with boundaries
        def get_grid_levels(self):
            if not self.use_caching:
                return self.__get_grid_levels()

            dict_key = self.__get_dict_key(self.grid)

            if self.__is_cached(self.grid):
                return self.__get_grid_levels_from_cache(dict_key)

            self.__cache_grid()
            return self.__get_grid_levels_from_cache(dict_key)

        def __get_grid_levels(self):
            return [0] + self.root_node.get_grid_levels() + [0]

        def __get_leafs(self):
            if self.root_node is None:
                return []

            nodes = self.root_node.get_nodes_using_dfs_in_order()

            leafs = []
            for node in nodes:
                if node.is_leaf():
                    leafs.append(node)

            return leafs

        # Caching
        def __is_cached(self, grid):
            dict_key = self.__get_dict_key(grid)

            return dict_key in self.full_tree_dict

        def __cache_grid(self):
            dict_key = self.__get_dict_key(self.grid)
            self.full_tree_dict[dict_key] = (self.__get_grid(), self.__get_grid_levels())

        def __get_grid_from_cache(self, dict_key):
            grid, _ = self.full_tree_dict[dict_key]

            return grid

        def __get_grid_levels_from_cache(self, dict_key):
            _, grid_levels = self.full_tree_dict[dict_key]

            return grid_levels

        @staticmethod
        def __get_dict_key(grid):
            return tuple(grid)

        class Node:
            def __init__(self, point, left_child=None, right_child=None,
                         parent=None, start_level=1,
                         print_debug=False):
                self.point = point
                self.left_child = left_child
                self.right_child = right_child
                self.parent = parent
                self.level = start_level if parent is None else parent.level + 1

                self.print_debug = print_debug

            def get_nodes_using_dfs_in_order(self, max_level=None):
                return self.__get_nodes_using_dfs_in_order_rec(self, max_level)

            def __get_nodes_using_dfs_in_order_rec(self, node, max_level=None):
                if node is None or (max_level is not None and node.level > max_level):
                    return []

                left_subtree = self.__get_nodes_using_dfs_in_order_rec(node.get_left_child(), max_level)
                right_subtree = self.__get_nodes_using_dfs_in_order_rec(node.get_right_child(), max_level)

                return left_subtree + [node] + right_subtree

            # Returns the grid of current subtree
            def get_grid(self, max_level=None):
                return list(map(lambda node: node.point, self.get_nodes_using_dfs_in_order(max_level)))

            # Returns the levels of the current subtree
            def get_grid_levels(self, max_level=None):
                return list(map(lambda node: node.level, self.get_nodes_using_dfs_in_order(max_level)))

            # Setter methods for children nodes
            def set_left_child(self, left_child):
                self.left_child = left_child
                left_child.level = self.level + 1

            def set_right_child(self, right_child):
                self.right_child = right_child
                right_child.level = self.level + 1

            # Getter methods
            def get_left_child(self):
                return self.left_child

            def get_right_child(self):
                return self.right_child

            def has_left_child(self):
                return self.get_left_child() is not None

            def has_right_child(self):
                return self.get_right_child() is not None

            def has_both_children(self):
                return self.has_left_child() and self.has_right_child()

            def has_only_one_child(self):
                return (self.has_left_child() and (not self.has_right_child())) \
                       or (self.has_right_child() and (not self.has_left_child()))

            def is_leaf(self):
                return (not self.has_left_child()) and (not self.has_right_child())

            def is_root_node(self):
                return self.parent is None

            def has_parent(self):
                return not self.is_root_node()

            def is_left_child(self):
                return self.parent.get_left_child() is self

            def is_right_child(self):
                return self.parent.get_right_child() is self

            @staticmethod
            def node_to_string(node):
                return "Node: Point {} of level {})".format(node.point, node.level)

    instance = None

    def __init__(self, print_debug=False):
        if not GridBinaryTree.instance:
            GridBinaryTree.instance = GridBinaryTree.__GridBinaryTree(print_debug=print_debug)
        else:
            GridBinaryTree.instance.print_debug = print_debug

    def __getattr__(self, name):
        return getattr(self.instance, name)

    # Interface methods
    def init_tree(self, grid, grid_levels):
        return self.instance.init_tree(grid, grid_levels)

    def force_full_tree_invariant(self):
        self.instance.force_full_tree_invariant()

    def init_perfect_tree_with_max_level(self, a, b, max_level):
        self.instance.init_perfect_tree_with_max_level(a, b, max_level)

    def increment_level_in_each_subtree(self):
        self.instance.increment_level_in_each_subtree()

    def get_grid(self):
        return self.instance.get_grid()

    def get_grid_levels(self):
        return self.instance.get_grid_levels()


# -----------------------------------------------------------------------------------------------------------------
# ---  Romberg Grid


class SliceGrouping(Enum):
    """
    This enum specifies the available options for slice grouping.
    """
    UNIT = 1  # Only unit slices (no grouping)
    GROUPED = 2  # Group slices if size 2^k, split into unit containers otherwise
    GROUPED_OPTIMIZED = 3   # Group slices until container size is the max possible 2^k


class SliceVersion(Enum):
    """
    This enum specifies the available slice types.
    """
    ROMBERG_DEFAULT = 1  # Sliced Romberg with default Romberg extrapolation
    ROMBERG_SLICED = 2  # Sliced Romberg with spliced trapezoidal extrapolation
    TRAPEZOID = 3  # Default trapezoidal rule without extrapolation


class SliceContainerVersion(Enum):
    """
    This enum specifies the available slice types.
    """
    ROMBERG_DEFAULT = 1  # Default Romberg extrapolation in non-unit containers


class RombergGrid:
    """
    This is the main extrapolation class. It initializes the containers and slices.
    Collects their (extrapolated) weights and returns the weights for each grid point.

    :param slice_grouping: Specifies which type of grouping is used for the slices (e.g. UNIT, or GROUPED).
    :param slice_version: This parameters specifies the type of extrapolation within a slice.
    :param container_version: Specifies the extrapolation type within a container.
    :param force_full_binary_tree_grid: If enabled, the provided grid will be extended to a full binary grid
            (each point, except the boundary points, has two children).
    :param print_debug: Print output to console.
    """
    def __init__(self,
                 slice_grouping: SliceGrouping = SliceGrouping.UNIT,
                 slice_version: SliceVersion = SliceVersion.ROMBERG_DEFAULT,
                 container_version: SliceContainerVersion = SliceContainerVersion.ROMBERG_DEFAULT,
                 force_full_binary_tree_grid=False,
                 print_debug=False):
        self.print_debug = print_debug

        self.a = None
        self.b = None
        self.grid = None
        self.grid_levels = None
        self.slice_containers = None
        self.weights = None
        self.function = None
        self.integral_approximation = None

        # Different grid versions
        self.slice_grouping = slice_grouping
        self.slice_version = slice_version
        self.container_version = container_version
        self.force_full_binary_tree_grid = force_full_binary_tree_grid

        # Factories
        self.slice_factory = ExtrapolationGridSliceFactory(self.slice_version)
        self.container_factory = ExtrapolationGridSliceContainerFactory(self.container_version)

    # Integration
    def integrate(self, function: Function = None):
        """
        This method integrates a given function over a grid that has been specified before.

        :param function: function to be integrated.
        :returns: the approximated integral.
        """

        assert self.grid is not None
        assert self.grid_levels is not None
        assert self.slice_containers is not None

        assert len(self.grid) >= 2

        # Update function and anti derivative
        if function is not None:
            self.set_function(function)

        # Pass the updated function to containers!
        self.update_function_in_containers()

        assert self.function is not None

        # Compute scalar product of weights and function values
        if self.weights is None:
            self.update_weights()

        assert len(self.weights) == len(self.grid), "Weight count and grid point count are different"

        value = 0
        for i in range(len(self.grid)):
            value += self.weights[i] * self.function.evaluate_at(self.grid[i])

        self.integral_approximation = value

        return value

    def set_grid(self, grid, grid_levels):
        """
        This method updates the grid and initializes the new slices.

        :param grid: The new grid points as array.
        :param grid_levels: The corresponding grid levels as array.
        :returns: void.
        """

        assert len(grid) == len(grid_levels) and len(grid) >= 2, "Wrong grid or grid levels provided."

        self.grid = grid
        self.grid_levels = grid_levels
        self.weights = None

        if self.force_full_binary_tree_grid:
            print("Forcing the grid to its closest full binary tree")
            tree = GridBinaryTree(print_debug=self.print_debug)
            tree.init_tree(grid, grid_levels)
            tree.force_full_tree_invariant()
            self.grid = tree.get_grid()
            self.grid_levels = tree.get_grid_levels()

        # Set boundary points
        self.a = self.grid[0]
        self.b = self.grid[-1]

        if self.print_debug:
            print("The grid is set to  \n   {}, with grid levels \n   {}".format(grid, grid_levels))

        # Initialize the new slices
        self.__init_grid_slices()

        if self.print_debug:
            print("The grid slices have been initialized \n")

    def __init_grid_slices(self):
        """
        This method partitions the integration area into slices (based on the provided grid).
        The slices are grouped into containers.

        :returns: void.
        """

        # Reset slice containers
        self.slice_containers = []

        # This buffer stores the step width of the previous grid slice
        step_width_buffer = None

        for i in range(len(self.grid) - 1):
            # Boundaries of this slice
            start_point = self.grid[i]
            start_level = self.grid_levels[i]
            end_point = self.grid[i + 1]
            end_level = self.grid_levels[i + 1]

            # Assert that the step width is h_k = 1/(2^k)
            step_width = end_point - start_point
            max_level = max(start_level, end_level)
            message = "The step width {} between x_{} and x_{} is not of the form 1/(2^k)".format(step_width, i, i + 1)
            assert step_width == self.get_step_width(max_level), message

            support_sequence = self.compute_support_sequence(i, i + 1)

            grid_slice = self.slice_factory.get_grid_slice(
                [start_point, end_point],
                [start_level, end_level],
                support_sequence,
                function=self.function
            )

            # Group  >= 1 slices into a container, which spans a partial full grid
            #   Create a new container...
            #   ... if this is the first iteration or
            #   ... if the step width changes or
            #   ... the version of this grid is defined to be unit slices (then each container holds only one slice)
            if (step_width_buffer is None) \
                    or (step_width != step_width_buffer) \
                    or (self.slice_grouping == SliceGrouping.UNIT):

                container = self.container_factory.get_grid_slice_container(function=self.function)
                container.append_slice(grid_slice)

                self.slice_containers.append(container)

            #   Append to previous container if the step width hasn't changed
            elif step_width == step_width_buffer:
                self.slice_containers[-1].append_slice(grid_slice)

            # Update buffer
            step_width_buffer = step_width

        # Split containers if necessary into unit slices (Each container should contain 2^k slices for k > 0)
        self.adjust_containers()

    def compute_support_sequence(self, final_slice_start_index, final_slice_stop_index):
        """
        This method computes the sequence of support points for the sliced trapezoid.

        :param final_slice_start_index: index of the left boundary in the grid array.
        :param final_slice_stop_index: index of the right boundary in the grid array.
        :returns: support sequence, e.g. [(0, 1), (0, 0.5), (0.25, 0.5)].
        """

        # Init sequence with indices for level 0 (whole integration domain)
        sequence = [(0, len(self.grid) - 1)] \
                   + self.__compute_support_sequence_rec(0, len(self.grid) - 1,
                                                         final_slice_start_index, final_slice_stop_index)

        return list(map(lambda element: (self.grid[element[0]], self.grid[element[1]]), sequence))

    def __compute_support_sequence_rec(self, start_index, stop_index,
                                       final_slice_start_index, final_slice_stop_index):
        """
        This method recursively computes the support sequence.

        :param start_index: current start index in array.
        :param stop_index: current stop index in array.
        :param final_slice_start_index: index of the left boundary in the grid array.
        :param final_slice_stop_index: index of the right boundary in the grid array.
        :returns: support sequence except the first support tuple, e.g. [(0, 0.5), (0.25, 0.5)].
        """

        if start_index >= stop_index:
            return []

        # Slice of the grid levels without the current level
        grid_levels_slice = self.grid_levels[(start_index + 1):stop_index]

        if len(grid_levels_slice) == 0:
            return []

        # Start or stop index of the next slice in the sequence
        new_boundary_index = (start_index + 1) + grid_levels_slice.index(min(grid_levels_slice))

        # Determine the boundary indices of the new slice in the sequence
        if new_boundary_index <= final_slice_start_index:
            start_index = new_boundary_index
            # stop_index does not change
        else:
            # start_index does not_change
            stop_index = new_boundary_index

        return [(start_index, stop_index)] + self.__compute_support_sequence_rec(start_index, stop_index,
                                                                                 final_slice_start_index,
                                                                                 final_slice_stop_index)

    def adjust_containers(self):
        """
        Split containers if necessary into unit slices. Each container should contain 2^k slices for k > 0.
        It updates the attributes accordingly.

        :returns: void.
        """

        new_containers = []

        for i, container in enumerate(self.slice_containers):
            size = container.size()

            # Split container if size != 2^k for k >= 0 (<=> log_2(size) isn't an integer)
            #   Split until size is the next closest power of 2
            if not (math.log(size, 2)).is_integer():
                # Find closest power of 2
                closest_power_below = self.__find_closest_power_below(size, 2)

                # Default grouping: Create new container for each slice
                if self.slice_grouping != SliceGrouping.GROUPED_OPTIMIZED:
                    for slice in container.slices:
                        container = self.container_factory.get_grid_slice_container(function=self.function)
                        container.append_slice(slice)
                        self.__assert_container_size(container)

                        new_containers.append(container)
                # Optimized container splitting. Split remainders slices until container size is 2^k
                else:
                    new_containers.append(container)
                    new_index = len(new_containers) - 1

                    for (j, slice) in enumerate(container.slices):
                        # Create new container for each slice whose index exceeds the closest power below (2^k)
                        if j >= closest_power_below:
                            container = self.container_factory.get_grid_slice_container(function=self.function)
                            container.append_slice(slice)
                            self.__assert_container_size(container)

                            new_containers.append(container)

                    # Remove slices from initial container that have been split into a new unit container
                    new_containers[new_index].slices = \
                        new_containers[new_index].slices[0:closest_power_below]
                    self.__assert_container_size(new_containers[new_index])
            else:
                new_containers.append(container)
                self.__assert_container_size(container)

        self.slice_containers = new_containers

    @staticmethod
    def __assert_container_size(container):
        """
        This method assures that the container has a size of 2^k (Assert).

        :param container: Extrapolation container with slices.
        :returns: void.
        """

        assert (math.log(container.size(), 2)).is_integer()

    @staticmethod
    def __find_closest_power_below(n, base=2):
        """
        This method determines the closest power below a given number for a given base and returns it.

        :param n: upper bound.
        :param base: base for exponentiation.
        :returns: closest 2^k below given number n.
        """
        for i in range(n):
            left_pow = math.pow(base, i)
            right_pow = math.pow(base, i + 1)

            if left_pow < n < right_pow:
                return int(left_pow)

        return 1

    def update_weights(self):
        """
        This method computes the weights for the current grid and overrides the old weights.

        :returns: void.
        """
        self.weights = self.get_weights()

    def get_weights(self):
        """
        Get extrapolated weights as array (sorted by grid points in increasing order).

        :returns: weights as array.
        """
        grid_weights = []
        weight_dictionary = self.__get_final_weights_from_all_slices()

        for _, weights in sorted(weight_dictionary.items()):
            grid_weights.append(sum(weights))

        return grid_weights

    def __get_final_weights_from_all_slices(self):
        """
        This method iterates over all containers and collects their weights in a single dictionary.

        :returns: all weights of the current grid as dictionary.
        """
        weight_dictionary = defaultdict(list)

        # Iterate over all grid slice containers and collect their weights into one dictionary
        for container in self.slice_containers:
            weight_dictionary_for_slice = container.get_final_weights()

            # Display error of this slice before and after extrapolation
            if self.print_debug and container.function is not None:
                print()
                container.print_error_evolution()
                print()

            for grid_point, weights in weight_dictionary_for_slice.items():
                weight_dictionary[grid_point].extend(weights)

        return weight_dictionary

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Setter

    def set_function(self, function):
        self.function = function

        self.update_function_in_containers()

    def update_function_in_containers(self):
        # Update function in slices
        if self.slice_containers is not None:
            for container in self.slice_containers:
                container.set_function(self.function)

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Getter

    def get_grid(self):
        return self.grid

    def get_grid_levels(self):
        return self.grid_levels

    def get_absolute_error(self):
        actual_result = self.integral_approximation

        if actual_result is None:
            actual_result = self.integrate()

        return abs(actual_result - self.function.get_analytical_solution(self.a, self.b))

    def get_step_width(self, level):
        return (self.b - self.a) / (2 ** level)

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Plot

    def plot_slices_with_function(self):
        """
        This method plots the function as well as the grid slices/containers.

        :returns: void.
        """
        assert self.function is not None
        grid = self.get_grid()

        x = np.array(grid)
        y = np.array([self.function.evaluate_at(xi) for xi in x])

        # X and Y values for plotting y=f(x)
        X = np.linspace(self.a, self.b, 100)
        Y = np.array([self.function.evaluate_at(xi) for xi in X])
        plt.plot(X, Y)

        for i in range(len(x) - 1):
            xs = [x[i], x[i], x[i + 1], x[i + 1]]
            plt.plot([x[i]], [self.function.evaluate_at(x[i])], marker='o', markersize=3, color="red")
            ys = [0, self.function.evaluate_at(x[i]), self.function.evaluate_at(x[i + 1]), 0]
            plt.fill(xs, ys, 'b', edgecolor='b', alpha=0.2)

        plt.plot([x[-1]], [self.function.evaluate_at(x[-1])], marker='o', markersize=3, color="red")
        # plt.xticks(list(range(len(grid))), grid)

        plt.text(x[-1] + 0.15, y[-1], "f")
        plt.title("RombergGrid Slices")

        plt.show()


class ExtrapolationGridSlice(ABC):
    """
    This abstract class provides an interface for grid slices.

    :param interval: An array that contains the two boundary points of this slice.
    :param levels: An array with the level of the left and right point.
    :param support_sequence: A sequence of refined support points for extrapolation.
    :param function: for error computation.
    """
    def __init__(self, interval, levels, support_sequence, function: Function = None):
        assert interval[0] < interval[1]

        self.left_point = interval[0]
        self.right_point = interval[1]
        self.width = self.right_point - self.left_point

        self.max_level = max(levels)
        self.levels = levels

        assert self.max_level == len(support_sequence) - 1
        self.support_sequence = support_sequence

        # Weights
        self.extrapolated_weights_dict = None

        # Analytical function for comparison
        self.function = None
        self.analytical_solution = None

        self.set_function(function)

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Weights

    @abstractmethod
    def get_weight_for_left_and_right_support_point(self, left_support_point, right_support_point):
        """
        This methods computes the weights for the support points, that correspond to the area of this slice

        :param left_support_point: left point from the current element of the support sequence
        :param right_support_point: right point from the current element of the support sequence
        :returns: (left_weight, right_weight).
        """

        pass

    def get_support_points_with_their_weights(self, level):
        """
        This method returns the left/right support point with its corresponding weight for a given level

        Example: Stripe [0, 0.5], Grid [0, 0.5, 0.625, 0.75, 1], Level 0.
        left support point is 0, right support point is 1!! (not 1/2)

        => returns [(0, weight_0), (1, weight_1)]

        :param level: This parameter determines the index of element from the support sequence.
                        Level -1 indexes the slice boundaries. Level 0 indexes the integration domain boundaries.
        :returns: (left_point, left_weight), (right_point, right_weight).
        """

        assert 0 <= level <= self.max_level

        left_point, right_point = self.support_sequence[level]
        left_weight, right_weight = self.get_weight_for_left_and_right_support_point(left_point, right_point)

        return [(left_point, left_weight), (right_point, right_weight)]

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Extrapolation of weights

    @abstractmethod
    def get_final_weights(self):
        """
        This method computes the final (extrapolated) weights.

        :returns: a dictionary that maps grid points to a list of their (extrapolated) weights.
        """

        pass

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Errors

    def get_absolute_error_at_level(self, level):
        """
        This method computes the absolute error at a given level

        :param level: index in the support sequence
        :returns: absolute error of this slice at a given level (in respect the the analytical solution).
                    This is not the final error.
        """
        assert self.function is not None
        point_weight_pairs = self.get_support_points_with_their_weights(level)

        slice_approximation_value = 0

        for (point, weight) in point_weight_pairs:
            slice_approximation_value += weight * self.function.evaluate_at(point)

        return abs(slice_approximation_value - self.analytical_solution)

    def get_extrapolated_error(self):
        """
        This method computes the absolute final error.

        :returns: absolute error (in respect the the analytical solution) of the extrapolated slice.
        """

        assert self.function is not None

        # Get extrapolated weights
        if self.extrapolated_weights_dict is None:
            self.extrapolated_weights_dict = self.get_final_weights()

        weight_dict = self.extrapolated_weights_dict

        # Compute value
        slice_approximation_value = 0
        for grid_point, weights in weight_dict.items():
            slice_approximation_value += sum(weights) * self.function.evaluate_at(grid_point)

        return abs(slice_approximation_value - self.analytical_solution)

    def print_error_evolution(self):
        # Print default error
        for level in range(self.max_level + 1):
            error = self.get_absolute_error_at_level(level)
            print("   {} on level {}".format(error, level))

        # Print extrapolated error
        extrapolated_error = self.get_extrapolated_error()
        print("   {} after extrapolation".format(extrapolated_error))

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Getter & Setter

    def set_function(self, function):
        self.function = function

        if function is not None:
            self.analytical_solution = function.get_analytical_solution(self.left_point, self.right_point)
        else:
            self.analytical_solution = None

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Helpers

    def to_string(self, name="ExtrapolationGridSlice"):
        return "{}} [{}, {}]".format(name, self.left_point, self.right_point)


class ExtrapolationGridSliceContainer(ABC):
    """
    This abstract class provides an interface for grid slice containers.
    Implementing classes store >= 1 grid slices that are adjacent and together span a partial full grid
    of equidistant step width.

    :param function: for error computation.
    """
    def __init__(self, function: Function = None):
        self.slices = []
        self.left_point = None
        self.right_point = None

        # Determination of step width by min and max level
        self.max_level = None  # This max_level is not shifted!!

        self.extrapolated_weights_dict = None

        # Analytical function for comparison
        self.function = None
        self.analytical_solution = None
        self.set_function(function)

    def append_slice(self, slice: ExtrapolationGridSlice):
        """
        This method appends a slice of arbitrary type to the container.

        :param slice: An extrapolation grid slice of arbitrary type.
        :returns: void.
        """

        # Update container interval
        if self.left_point is None:
            self.left_point = slice.left_point

        # Right bound has to be updated every time a new slice is appended
        self.right_point = slice.right_point
        self.max_level = max(self.max_level, slice.max_level) if self.max_level is not None else slice.max_level

        self.slices.append(slice)

    @abstractmethod
    def get_final_weights(self):
        """
        This method computes the final (extrapolated) slices in this container.

        :returns: a dictionary of weights.
        """

        pass

    def get_grid(self):
        """
        This method returns all grid points in this container.

        :returns: the grid of this container as array
        """

        return list(map(lambda s: s.left_point, self.slices)) + [self.slices[-1].right_point]

    def get_grid_levels(self):
        """
        This method returns all grid levels in this container.

        :returns: the grid levels of this container as array
        """

        return list(map(lambda s: s.levels[0], self.slices)) + [self.slices[-1].levels[1]]

    def get_normalized_grid_levels(self):
        """
        This method normalizes the grid levels
        E.g. Given a full Romberg grid [0, 0.5, 0.625, 0.75, 1]
                             with levels [0, 1, 3, 2, 0]
        The container grid is [0.5, 0.625, 0.75]
             with grid levels [1, 3, 2]

        => normalized grid levels [0, 1, 0].

        :returns: the normalized grid levels of this container.
        """

        grid = self.get_grid()

        # Unit slice grid needs no normalization
        if len(grid) == 2:
            return self.get_grid_levels()

        # The container grid must contain and odd number of grid points
        assert len(grid) % 2 == 1
        assert len(grid) >= 3
        # Assert that the there are 2^k slices
        self.__assert_size()

        return [0] + self.__get_normalized_grid_levels(1, len(grid) - 2) + [0]

    def __get_normalized_grid_levels(self, start, stop, level=1):
        middle = int((start + stop) / 2)

        if start > stop:
            return []
        elif start == stop:
            return [level]

        left_levels = self.__get_normalized_grid_levels(start, middle - 1, level + 1)
        right_levels = self.__get_normalized_grid_levels(middle + 1, stop, level + 1)

        return left_levels + [level] + right_levels

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Errors

    def get_extrapolated_error(self):
        """
        Computes the absolute final error.

        :returns: absolute extrapolated error to the analytical solution
        """

        assert self.function is not None

        # Get extrapolated weights
        if self.extrapolated_weights_dict is None:
            self.extrapolated_weights_dict = self.get_final_weights()

        weight_dict = self.extrapolated_weights_dict

        # Compute value
        slice_approximation_value = 0
        for grid_point, weights in weight_dict.items():
            slice_approximation_value += sum(weights) * self.function.evaluate_at(grid_point)

        return abs(slice_approximation_value - self.analytical_solution)

    def print_error_evolution(self, name="SliceContainer"):
        print("The {} [{}, {}] has the following error evolution:".format(name, self.left_point, self.right_point))

        # Print unit slice error evolution
        if len(self.slices) == 1:
            self.slices[0].print_error_evolution()
            return

        print("   {} after extrapolation".format(self.get_extrapolated_error()))

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Getter & Setter

    def set_function(self, function):
        self.function = function

        if function is not None and self.left_point is not None and self.right_point is not None:
            self.analytical_solution = function.get_analytical_solution(self.left_point, self.right_point)
        else:
            self.analytical_solution = None

        # Update function in all slices
        for slice in self.slices:
            slice.set_function(function)

    def get_slices(self):
        return self.slices

    # Returns the amount of slices in this container
    def size(self):
        return len(self.slices)

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Helpers

    def to_string(self, name="SliceContainer"):
        str_builder = "{} [{}, {}] with the slices: \n".format(name, self.left_point, self.right_point)

        for i, slice in enumerate(self.slices):
            str_builder += "   " + slice.to_string() + "\n"

        return str_builder

    def __assert_size(self):
        assert (math.log(self.size(), 2)).is_integer()


class RombergGridSlice(ExtrapolationGridSlice):
    """
    This class store a grid slice for extrapolation.

    :param interval: An array that contains the two boundary points of this slice.
    :param levels: An array with the level of the left and right point.
    :param support_sequence: A sequence of refined support points for extrapolation.
    :param function: for error computation.
    """
    def __init__(self, interval, levels, support_sequence,
                 extrapolation_version: ExtrapolationVersion, function: Function = None):
        super(RombergGridSlice, self).__init__(interval, levels, support_sequence, function)

        self.extrapolation_version = extrapolation_version
        self.coefficient_factory = ExtrapolationCoefficientsFactory(self.extrapolation_version)

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Weights

    # Based on generalized sliced trapezoidal rule
    # See thesis, for the derivation of this weights
    # This methods computes the weights for the support points, that correspond to the area of this slice
    def get_weight_for_left_and_right_support_point(self, left_support_point, right_support_point):
        # Support points may only be located outside of this slice, or on the boundaries itself
        assert left_support_point <= self.left_point and right_support_point >= self.right_point
        assert left_support_point != right_support_point

        # Compute common terms for the left and right weight
        slice_width = self.width

        support_point_ratio = left_support_point / (left_support_point - right_support_point)  # denominator always != 0
        slice_support_ratio = (1 / 2) * (
                (self.right_point + self.left_point) / (left_support_point - right_support_point)
        )

        # Determine the weight for the left support point
        left_weight = slice_width * (1 - support_point_ratio + slice_support_ratio)

        # Determine the weight for the right support point
        right_weight = slice_width * (support_point_ratio - slice_support_ratio)

        # Return the weight for the support point
        return left_weight, right_weight

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Extrapolation of weights

    # This method returns a dictionary that maps grid points to a list of their extrapolated weights
    def get_final_weights(self):
        # The first element of the support sequence contains the integration domain boundaries
        (a, b) = self.support_sequence[0]

        # Generate the extrapolation coefficients
        coefficient_factory = self.coefficient_factory.get(a, b, self.support_sequence)

        # Dictionary that maps grid points to their extrapolated weights
        weight_dictionary = defaultdict(list)

        for level in range(self.max_level + 1):  # 0 <= i <= max_level
            point_weight_pairs = self.get_support_points_with_their_weights(level)

            assert len(point_weight_pairs) == 2
            (left_point, left_weight) = point_weight_pairs[0]
            (right_point, right_weight) = point_weight_pairs[1]

            coefficient = coefficient_factory.get_coefficient(self.max_level, level)
            left_weight = coefficient * left_weight
            right_weight = coefficient * right_weight

            weight_dictionary[left_point].append(left_weight)
            weight_dictionary[right_point].append(right_weight)

        # Update dictionary of extrapolated weights
        self.extrapolated_weights_dict = weight_dictionary

        return weight_dictionary

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Helpers

    def to_string(self, name=None):
        return super().to_string("RombergGridSlice")


# This class stores >= 1 grid slices that are adjacent and together span a partial full grid
# of equidistant step width
class RombergGridSliceContainer(ExtrapolationGridSliceContainer):
    """
    This class groups >= 1 grid slices that are adjacent and span a full grid of equidistant step width together.
    If the slice count is > 1 a (default or custom) Romberg is executed in this full grid.
    Else the unit slice is extrapolated based on its own extrapolation type.

    :param function: for error computation.
    """
    def __init__(self, function: Function = None):
        super(RombergGridSliceContainer, self).__init__(function)

    def get_final_weights(self):
        assert len(self.slices) > 0

        # This container has only one slice. => Extrapolate this unit slice
        if len(self.slices) == 1:
            return self.slices[0].get_final_weights()

        # This container has >= 2 slices
        # Execute default Romberg on this container
        factory = RombergWeightFactory(self.left_point, self.right_point, ExtrapolationVersion.ROMBERG_DEFAULT)
        grid = self.get_grid()
        normalized_grid_levels = self.get_normalized_grid_levels()
        normalized_max_level = max(normalized_grid_levels)

        weight_dictionary = defaultdict(list)

        # Extrapolate weights on this container
        for i, point in enumerate(grid):
            if (i == 0) or (i == len(grid) - 1):
                weight = factory.get_boundary_point_weight(normalized_max_level)
            else:
                weight = factory.get_inner_point_weight(normalized_grid_levels[i], normalized_max_level)

            weight_dictionary[point].append(weight)

        # Update pointer to dictionary
        self.extrapolated_weights_dict = weight_dictionary

        return weight_dictionary

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Helpers

    # See parent class
    def to_string(self, name=None):
        return super().to_string("RombergGridSliceContainer")


class TrapezoidalGridSlice(ExtrapolationGridSlice):
    """
     This type of GridSlice executes no extrapolation. It produces default trapezoidal weights.

     :param interval: An array that contains the two boundary points of this slice.
     :param levels: An array with the level of the left and right point.
     :param support_sequence: A sequence of refined support points for extrapolation.
     :param function: for error computation.
     """
    def __init__(self, interval, levels, support_sequence, function: Function = None):
        super(TrapezoidalGridSlice, self).__init__(interval, levels, support_sequence, function)

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Weights

    # See thesis, for the derivation of this weights
    # This methods computes the weights for the support points, that correspond to the area of this slice
    def get_weight_for_left_and_right_support_point(self, left_support_point, right_support_point):
        return self.width/2, self.width/2

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Final weights

    def get_final_weights(self):
        # Dictionary that maps grid points to their extrapolated weights
        weight_dictionary = defaultdict(list)

        # There is no Extrapolation happening in this type of grid slice
        left_weight, right_weight = \
            self.get_weight_for_left_and_right_support_point(left_support_point=self.left_point,
                                                             right_support_point=self.right_point)

        weight_dictionary[self.left_point].append(left_weight)
        weight_dictionary[self.right_point].append(right_weight)

        # Update dictionary of weights
        self.extrapolated_weights_dict = weight_dictionary

        return weight_dictionary

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Helpers

    def to_string(self, name=None):
        return super().to_string("TrapezoidalGridSlice")


class ExtrapolationGridSliceFactory:
    """
     This class encapsulates the creation of different grid slice types (Factory Pattern).

     :param slice_version: Determines which type of slice has to be created.
     """
    def __init__(self, slice_version: SliceVersion):
        self.slice_version = slice_version

    def get_grid_slice(self, interval, levels, support_sequence, function: Function = None) -> ExtrapolationGridSlice:
        if self.slice_version == SliceVersion.ROMBERG_DEFAULT:
            return RombergGridSlice(interval, levels, support_sequence,
                                    ExtrapolationVersion.ROMBERG_DEFAULT, function)

        elif self.slice_version == SliceVersion.ROMBERG_SLICED:
            return RombergGridSlice(interval, levels, support_sequence,
                                    ExtrapolationVersion.ROMBERG_SLICED, function)

        elif self.slice_version == SliceVersion.TRAPEZOID:
            return TrapezoidalGridSlice(interval, levels, support_sequence, function)

        else:
            raise RuntimeError("Wrong SliceVersion provided.")


class ExtrapolationGridSliceContainerFactory:
    """
     This class encapsulates the creation of different grid slice container types (Factory Pattern).

     :param slice_container_version: Determines which type of slice container has to be created.
     """
    def __init__(self, slice_container_version: SliceContainerVersion):
        self.slice_container_version = slice_container_version

    def get_grid_slice_container(self, function: Function = None) -> ExtrapolationGridSliceContainer:
        if self.slice_container_version == SliceContainerVersion.ROMBERG_DEFAULT:
            return RombergGridSliceContainer(function)
        else:
            raise RuntimeError("Wrong ContainerVersion provided.")
