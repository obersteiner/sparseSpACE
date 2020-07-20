from numpy import linalg as LA
from math import isclose, isinf
from Grid import *
from BasisFunctions import *
from RefinementContainer import RefinementContainer
from RefinementObject import RefinementObject

import time

class GridOperation(object):
    """This class defines the basic interface for a GridOperation which performs operations on a component grid.
    It should provide the basic functionalities to solve the operation on arbitrary grids and calculate error estimates
    for refinement. It also needs to provide functionality in how to combine the individual results on the component
    grids.
    """

    def is_area_operation(self) -> bool:
        """This function returns a bool to indicate whether the operation can be seperately applied to subareas.

        :return: Bool if function can be separately applied to subareas.
        """
        return False

    @abc.abstractmethod
    def perform_operation(self, levelvector, refinementObjects=None):
        """Main function of the GridOperation which applies the operation to the specified levelvector.

        :param levelvector:
        :param refinementObjects:
        :return:
        """
        pass

    @abc.abstractmethod
    def set_errors(self, refinementObjects=None):
        pass

    @abc.abstractmethod
    def count_unique_points(self) -> bool:
        """This method indicates whether we should only count unique points for the cost estimate (return True) or
        if we should count points multiple times if they are contained in different component grids

        :return: Bool
        """
        return False

    def area_preprocessing(self, area) -> None:
        """This area is used to preprocess an area before the operation starts.

        :param area: Area to preprocess.
        :return: None
        """
        pass

    def area_postprocessing(self, area) -> None:
        """This area is used to postprocess an area after the operation finishes.

        :param area: Area to preprocess.
        :return: None
        """
        pass

    def post_processing(self) -> None:
        """This method is used to postprocess after the operation has been applied to all component grids.

        :return: None
        """
        pass

    # This method can be used if there is a better global error estimate than the summation of the local surplusses
    def get_global_error_estimate(self, refinement: RefinementContainer) -> float:
        """This method returns the global error estimate. Should be implemented if there is a better estimator than sum
        of local errors in refinement.

        :param refinement: RefinementContainer from which to extract info for global error estimates.
        :return: Global error estimate
        """
        return None

    def get_grid(self) -> Grid:
        """This method returns the grid that is used by the GridOperation

        :return: Grid class
        """
        return self.grid

    def get_reference_solution(self) -> Sequence[float]:
        """This method returns the reference solution if available.

        :return: Reference solution.
        """
        return self.reference_solution

    def initialize(self) -> None:
        """This method can be used to initialize the GridOperation at the start of the adaptive procedure.

        :return: None
        """
        pass

    def compute_difference(self, first_value: Sequence[float], second_value: Sequence[float], norm) -> float:
        """This method calculates the difference measure (e.g error measure) between the combi result and the reference
        solution as a scalar value. Can be changed by Operation.

        :param first_value: Value that you want to compare to second value.
        :param second_value: Value that you want to compare to first value.
        :param norm: Norm in which the error should be calculated.
        :return: Difference measure
        """
        return LA.norm(abs(first_value - second_value), norm)

    def add_values(self, first_value: Sequence[float], second_value: Sequence[float]) -> Sequence[float]:
        """This value adds to values from GridOperation together and returns result.

        :param first_value: First value that you want to add up.
        :param second_value: Second value that you want to add up.
        :return: Addition of both values.
        """
        return first_value + second_value

    def get_result(self):
        """This method is called after the combination and should return the combination result.

        :return: Result of the combination.
        """
        pass

    def point_output_length(self) -> int:
        """Returns the length of list/array the model evaluations have at each point; in case of scalar values
        (e.g. of a scalar-valued function) it is 1, otherwise the vector length of the vector output.

        :return: Length of model values.
        """
        return 1

    def interpolate_points(self, values: Sequence[Sequence[float]], mesh_points_grid: Sequence[Sequence[float]],
                           evaluation_points: Sequence[Tuple[float, ...]]):
        """Interpolates values that are on the mesh_points_grid at the given evaluation_points using bilinear
        interpolation.

        :param values: Numpy array with values at grid points. Each value is again a numpy array.
        :param mesh_points_grid: Grid definition where values are placed. List of !D arrays.
        :param evaluation_points: Points at which we want to evaluate. List of points.
        :return:
        """
        return Interpolation.interpolate_points(values, self.dim, self.grid, mesh_points_grid, evaluation_points)

    @abc.abstractmethod
    def eval_analytic(self, coordinate: Tuple[float, ...]) -> Sequence[float]:
        """This method evaluates the analytic model at the given coordinate.

        :param coordinate: Coordinate for evaluation.
        :return: Value at coordinate.
        """
        pass

    @abc.abstractmethod
    def get_distinct_points(self):
        """This method returns the number of all points that were used in the combination.

        :return: Number of points.
        """
        pass

    @abc.abstractmethod
    def get_component_grid_values(self, component_grid: ComponentGridInfo, mesh_points_grid: Sequence[Sequence[float]]) -> Sequence[Sequence[float]]:
        """This method returns the grid values for the specified component grid on the specified mesh.

        :param component_grid: ComponentGridInfo of component grid for which we want the values.
        :param mesh_points_grid: Grid definition of the points at which we want the values.
        :return: List of all grid values. This is a 1D array with the values (each values is a numpy array!).
        """
        pass

    @abc.abstractmethod
    def evaluate_levelvec(self, component_grid: ComponentGridInfo) -> None:
        """This method evaluates the operation on a specified component grid based on the level vector. This method is
        used in the standard or dimension adaptive combination where we do not apply spatial adaptivity.

        :param component_grid: ComponentGridInfo of the specified component grid.
        :return:
        """
        pass

    def calculate_operation_dimension_wise(self, gridPointCoordsAsStripes: Sequence[Sequence[float]], grid_point_levels: Sequence[Sequence[int]],
                                           component_grid: ComponentGridInfo) -> None:
        """This method is used to compute the operation in the dimension-wise refinement strategy.

        :param gridPointCoordsAsStripes: Gridpoints as list of 1D lists
        :param grid_point_levels: Grid point levels as list of 1D lists
        :param component_grid: Component grid on which operation should be applied.
        :return: None
        """

    @abc.abstractmethod
    def compute_error_estimates_dimension_wise(self, gridPointCoordsAsStripes: Sequence[Sequence[float]], grid_point_levels: Sequence[Sequence[int]],
                                               children_indices: Sequence[Sequence[int]], component_grid: ComponentGridInfo) -> None:
        """This method is used to compute the error estimates in the dimension-wise refinement strategy.

        :param gridPointCoordsAsStripes: Gridpoints as list of 1D lists
        :param grid_point_levels: Grid point levels as list of 1D lists
        :param children_indices: List of children for each dimensions (list of lists)
        :param component_grid: Component grid on which operation should be applied.
        :return: None
        """
        pass

    @abc.abstractmethod
    def process_removed_objects(self, removed_objects: List[RefinementObject]) -> None:
        """This method is used whenever the refinement structure changes and contributions from old RefinementObjects
        need to be removed.

        :param removed_objects: RefinementObjects that were removed.
        :return: None
        """
        pass

    @abc.abstractmethod
    def get_point_values_component_grid(self, points, component_grid) -> Sequence[Sequence[float]]:
        """This method returns the values in the component grid at the given points.

        :param points: Points where we want to evaluate the componenet grid (should coincide with grid points)
        :param component_grid: Component grid which we want to evaluate.
        :return: Values at points (same order).
        """
        pass

    def get_surplus_width(self, d: int, right_parent: float, left_parent: float) -> float:
        """This method calculates the 1D surplus width for a linear basis function with left and right parent.

        :param d: Dimension we are in.
        :param right_parent: Right parent of the point (end of support)
        :param left_parent: Left parent of the point (beginning of support)
        :return:
        """
        return right_parent - left_parent


class AreaOperation(GridOperation):
    def is_area_operation(self):
        return True

    def perform_operation(self, levelvector, refinementObjects=None):
        for area in enumerate(refinementObjects):
            self.evaluate_area(area, levelvector)

    @abc.abstractmethod
    def evaluate_area(self, area, levelvector, componentgrid_info, refinement_container):
        pass


from numpy.linalg import solve
from scipy.sparse.linalg import cg
from scipy.integrate import nquad
from sklearn import datasets, preprocessing
import csv
from matplotlib import cm
import matplotlib.colors as colors


class DensityEstimation(AreaOperation):

    def __init__(self, data, dim, grid=None, masslumping=False, print_output=False, lambd=0.0, classes=None, reuse_old_values=False, numeric_calculation=False):
        self.data = data
        self.dim = dim
        if grid is None:
            self.grid = TrapezoidalGrid(a=np.zeros(self.dim), b=np.ones(self.dim), boundary=False)
        else:
            self.grid = grid
        self.lambd = lambd
        self.masslumping = masslumping
        self.surpluses = {}
        self.initialized = False
        self.scaled = False
        self.extrema = None
        self.print_output = print_output
        self.reference_solution = None
        self.debug = False
        self.classes = classes
        self.reuse_old_values = reuse_old_values
        self.old_R = {}  # for reuse_old_values
        self.old_B = {}  # for reuse_old_values
        self.new_B = {}  # for reuse_old_values
        self.old_grid_coord = {}  # for reuse_old_values; keys are a list of the max levels for each dimension
        self.new_grid_coord = {}
        self.sorted_data = []  # for reuse_old_values; keys are the dimensions
        self.data_bins = [{} for d in range(dim)]
        self.max_levels = []
        self.numeric_calculation = numeric_calculation
        print('DensityEstimation debug: ', self.debug)

    def initialize(self):
        """
        This method is used to initialize the operation with the dataset.
        If a path to a .csv file was specified, it gets read in and scaled to the intervall (0,1)
        It gets called in the perform_operation function of StandardCombi
        :return:
        """
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        if (isinstance(self.data, str)):
            dataCSV = []
            with open(self.data, "r", newline="") as file:
                has_header = csv.Sniffer().has_header(file.read(2048))
                file.seek(0)
                reader = csv.reader(file)
                if has_header:
                    next(reader)
                for row in reader:
                    dataCSV.append([float(i) for i in row])
                scaler.fit(dataCSV)
                if (any([(x < 0) for x in scaler.data_min_])) or (any([(x > 1) for x in scaler.data_max_])):
                    self.data = scaler.transform(dataCSV)
                    self.scaled = True
        elif (isinstance(self.data, tuple)):
            self.data = self.data[0]
            scaler.fit(self.data)
            if (any([(x < 0) for x in scaler.data_min_])) or (any([(x > 1) for x in scaler.data_max_])):
                self.data = scaler.transform(self.data)
                self.scaled = True
        else:
            scaler.fit(self.data)
            if (any([(x < 0) for x in scaler.data_min_])) or (any([(x > 1) for x in scaler.data_max_])):
                self.data = scaler.transform(self.data)
                self.scaled = True
        self.initialized = True

    def post_processing(self):
        """
        This method is used to compute the minimum and maximum surplus of the component grid
        so they can be used when plotting the heat map for the combi scheme when calling print_resulting_combi_scheme
        It gets called in the perform_operation function of StandardCombi
        :return: Tuple of minimum and maximum surplus
        """
        if self.reuse_old_values:
            # copy values, not references
            self.old_B = {}
            self.old_grid_coord = {}
            for key in self.new_B.keys():
                self.old_B[key] = list(self.new_B[key])
            for key in self.new_grid_coord.keys():
                self.old_grid_coord[key] = list(self.new_grid_coord[key])

            self.new_B = {}
            self.new_grid_coord = {}
            #self.create_new_data_bins()

        surpluses = np.concatenate(list(self.get_result().values()))
        max = np.max(surpluses)
        min = np.min(surpluses)
        if self.print_output:
            print("Max: ", max, "Min: ", min)
        self.extrema = (min, max)
        return self.extrema

    # def create_new_data_bins(self):
    #     if len(self.max_levels) == 0:
    #         for l in range(1, max(self.lmax)):
    #             for d in range(self.dim):
    #                 if d in self.data_bins.keys():
    #                     bins = self.data_bins[d]
    #                 else:
    #                     bins = {}
    #                 max_level = l
    #                 step_size = 1.0 / 2 ** max_level
    #                 lower = 0.0
    #                 upper = step_size
    #
    #                 while upper <= 1.0:
    #                     indices = self.sorted_data[d]
    #                     start_index = enclosing_bin[0] if enclosing_bin is not None else 0
    #                     end_index = enclosing_bin[1] if enclosing_bin is not None else len(indices)
    #
    #                     infimum = len(indices)
    #                     supremum = 0
    #                     # find the highest and lowest values' index within the range
    #                     for i in range(start_index, end_index):
    #                         if self.data[indices[i]][d] >= lower and i < infimum:
    #                             infimum = i
    #                         if i > supremum and self.data[indices[i]][d] <= upper:
    #                             supremum = i
    #                     bin = (infimum, supremum)
    #                     bins[str([lower, upper])] = bin
    #                     lower += step_size
    #                     upper += step_size
    #                 self.data_bins[d] = bins
    #     else:
    #         for d in range(self.dim):
    #             if d in self.data_bins.keys():
    #                 bins = self.data_bins[d]
    #             else:
    #                 bins = {}
    #             max_level = self.max_levels[d]
    #             step_size = 1.0 / 2 ** max_level
    #             lower = 0.0
    #             upper = step_size
    #
    #             while upper <= 1.0:
    #                 # check if we have an enclosing bin, so we don't have to search the entire data set
    #                 enclosing_bin = None
    #                 extract_list = lambda x: list(map(float, x[1:-1].split(',')))
    #                 if len(bins) > 0:
    #                     enclosing_bin_keys = [extract_list(key_string) for key_string in bins.keys()]
    #                     enclosing_bins = [x for x in enclosing_bin_keys if x[0] <= lower and x[1] >= upper]
    #                     if len(enclosing_bins) > 0:
    #                         enclosing_bin = bins[str(enclosing_bins[0])]
    #
    #                 indices = self.sorted_data[d]
    #                 start_index = enclosing_bin[0] if enclosing_bin is not None else 0
    #                 end_index = enclosing_bin[1] if enclosing_bin is not None else len(indices)
    #
    #                 infimum = len(indices)
    #                 supremum = 0
    #                 # find the highest and lowest values' index within the range
    #                 for i in range(start_index, end_index):
    #                     if self.data[indices[i]][d] >= lower and i < infimum:
    #                         infimum = i
    #                     if i > supremum and self.data[indices[i]][d] <= upper:
    #                         supremum = i
    #                 bin = (infimum, supremum)
    #                 bins[str([lower, upper])] = bin
    #                 lower += step_size
    #                 upper += step_size
    #             self.data_bins[d] = bins

    def get_result(self) -> Dict[Sequence[int], Sequence[float]]:
        return self.surpluses

    def get_reference_solution(self) -> None:
        return None

    def get_distinct_points(self):
        s = sum([len(x) for x in self.surpluses.values()])
        return s

    def evaluate_levelvec(self, component_grid: ComponentGridInfo) -> Sequence[float]:
        """
        This method calculates the surpluses for the the specified component grid
        :param component_grid: ComponentGridInfo of the specified component grid
        :return: Surpluses of the component grid
        """
        self.grid.setCurrentArea(np.zeros(len(component_grid.levelvector)), np.ones(len(component_grid.levelvector)), component_grid.levelvector)
        surpluses = self.solve_density_estimation(component_grid.levelvector)
        self.surpluses.update({tuple(component_grid.levelvector): surpluses})
        return surpluses

    def calculate_operation_dimension_wise(self, gridPointCoordsAsStripes: Sequence[Sequence[float]],
                                           grid_point_levels: Sequence[Sequence[int]], component_grid: ComponentGridInfo):
        """This method is used to compute the operation in the dimension-wise refinement strategy.

        :param gridPointCoordsAsStripes: Gridpoints as list of 1D lists
        :param grid_point_levels: Grid point levels as list of 1D lists
        :param component_grid: Component grid on which operation should be applied.
        :return: None
        """
        # TODO
        reuse_old_values = False
        # if reuse_old_values:
        #     whole bunch of stuff goes here when implemented
        # else:
        self.grid_surplusses.set_grid(gridPointCoordsAsStripes, grid_point_levels)
        self.grid.set_grid(gridPointCoordsAsStripes, grid_point_levels)
        surpluses = self.solve_density_estimation_dimension_wise(gridPointCoordsAsStripes, grid_point_levels, component_grid)

        #self.refinement_container.value += np.array(LA.norm(abs(surpluses), 2)) * component_grid.coefficient
        self.refinement_container.value += np.array(abs(surpluses.sum() / surpluses.size)) * component_grid.coefficient
        self.surpluses.update({tuple(component_grid.levelvector): surpluses * component_grid.coefficient})

        #if reuse_old_values:
        #    # do nothing as of yet
        #    True

    def init_dimension_wise(self, grid, grid_surplusses, refinement_container, lmin, lmax, a, b, version=2):
        self.grid = grid
        self.grid_surplusses = grid_surplusses
        self.refinement_container = refinement_container
        self.version = version
        self.lmin = lmin
        self.lmax = lmax
        self.a = a
        self.b = b

    def initialize_evaluation_dimension_wise(self, refinement_container):
        # TODO
        refinement_container.value = np.zeros(1)
        self.sorted_data = [np.argsort(self.data[:,d]) for d in range(self.data.shape[1])]
        #if self.reuse_old_values:
        #    self.create_new_data_bins()
        self.max_levels = [max(self.lmax) for x in range(self.dim)]

    # def get_existing_indices(self, levelvec):
    #     """
    #     This method returns a list of existing index tuples for the grid based on the coordinate_array of the grid
    #     :return: List of index tuples
    #     """
    #     full_index_list = get_cross_product_list([range(1, n + 1) for n in self.grid.levelToNumPoints(levelvec)])
    #     # points = self.grid.levelToNumPoints(levelvec)
    #     coords = self.grid.coordinate_array
    #     boundary = int(self.grid.boundary)
    #     # filter indices whose coordinates are not in the coordinate array
    #     index_list = list(filter(lambda x: ((x[d] * (1 / 2 ** levelvec[d])) - boundary * (1 / 2 ** levelvec[d]) in coords[d] for d in enumerate(levelvec)), full_index_list))
    #     return index_list

    def get_component_grid_values(self, component_grid: ComponentGridInfo, mesh_points_grid: Sequence[Sequence[float]]) -> Sequence[float]:
        """
        This method fills up the surplus array with zeros for the points on the boundary so it can be properly used when interpolating
        :param component_grid: ComponentGridInfo of the specified component grid
        :param mesh_points_grid: Points of the component grid, with boundary points
        :return: Surpluses for the component_grid filled up with zero on the boundary
        """
        surpluses = list(self.get_result().get(tuple(component_grid.levelvector)))
        mesh_points = get_cross_product(mesh_points_grid)
        values = np.array([surpluses.pop(0) if self.grid.point_not_zero(p) else 0 for p in mesh_points])
        return values.reshape((len(values), 1))

    def get_point_values_component_grid(self, points: Sequence[float], component_grid: ComponentGridInfo) -> Sequence[Sequence[float]]:
        # TODO
        """This method returns the values in the component grid at the given points.

        :param points: Points where we want to evaluate the componenet grid (should coincide with grid points)
        :param component_grid: Component grid which we want to evaluate.
        :return: Values at points (same order).
        """
        if self.debug:
            mesh_points_grid = [self.grid.coordinate_array[d] for d in range(self.dim)]
            mesh_points = list(get_cross_product(mesh_points_grid))
            points_list = list(points)
            points_indices = [mesh_points.index(p) for p in points_list]
            nodal_values = list(self.get_result().get(tuple(component_grid.levelvector)))
            if len(nodal_values) > 0 and len(points_list) > 0:
                pickPoint = lambda x: nodal_values[mesh_points.index(x)] if self.grid.point_not_zero(x) else 0
                validPoint = lambda x: True if x in mesh_points else False
                values = np.array([pickPoint(p) for p in points_list if validPoint(p)])
                values_indices = [nodal_values.index(s) for s in values]
                if (len(nodal_values) == 0 or len(points_list) == 0 or len(values) == 0):
                    print('Operation DensityEstimation; error, stop here with debugger')
                    print('second print to prevent the stupid debugger from skipping lines again')
                #if points_indices != values_indices:
                #    # if nodal_values has duplicate entries we can also end up here
                #    print('returned wrong component grid value(s)')
                return values.reshape((len(values), 1))
                #return values
            else:
                return np.zeros((len(points_list), 1))
                #return np.zeros(len(points_list))
        else:
            mesh_points_grid = (self.grid.coordinate_array[d] for d in range(self.dim))
            mesh_points = list(get_cross_product(mesh_points_grid))
            nodal_values = self.get_result().get(tuple(component_grid.levelvector))
            pickPoint = lambda x: nodal_values[mesh_points.index(x)] if self.grid.point_not_zero(x) else 0
            validPoint = lambda x: True if x in mesh_points else False
            values = np.array([pickPoint(p) for p in points if validPoint(p)])
            return values.reshape((len(values), 1))

    def check_adjacency(self, ivec: Sequence[int], jvec: Sequence[int]) -> bool:
        """
        This method checks if the two hat functions specified by ivec and jvec are adjacent to each other
        :param ivec: Index of the first hat function
        :param jvec: Index of the second hat function
        :return: True if the two hat functions are adjacent, False otherwise
        """
        for i in range(len(ivec)):
            if abs(ivec[i] - jvec[i]) > 1:
                return False
        return True

    def get_hats_in_support(self, levelvec: Sequence[int], x: Sequence[float]) -> Sequence[Tuple[int, ...]]:
        """
        This method returns all the hat functions in whose support the data point x lies
        :param levelvec: Levelvector of the component grid
        :param x: datapoint
        :return: All the hat functions in whose support the data point x lies
        """
        if self.grid.point_not_zero(x) and ((x >= 0).all() and (x <= 1).all()):
            meshsize = [2 ** (-float(list(levelvec)[d])) for d in range(self.dim)]
            numb_points = self.grid.levelToNumPoints(levelvec)
            index_set = []
            for i in range(len(x)):
                lower = math.floor(x[i] / meshsize[i])
                upper = math.ceil(x[i] / meshsize[i])
                if (lower > 0 and lower <= numb_points[i]) and (upper > 0 and upper <= numb_points[i]):
                    index_set.append((lower, upper))
                elif (lower < 1 or lower > numb_points[i]):
                    index_set.append((upper,))
                elif (upper < 1 or upper > numb_points[i]):
                    index_set.append((lower,))
            #test = list(set(product(*index_set)))
            #if not self.grid.is_global():
            return list(set(product(*index_set)))
            #else:
            #    existing_indices = self.get_existing_indices(levelvec)
            #    result = list(filter(lambda p: p in existing_indices, list(set(product(*index_set)))))
            #    return result
        else:
            return []

    def get_neighbors(self, point: Sequence[float], gridPointCoordsAsStripes: Sequence[Sequence[float]]) -> Sequence[Tuple[float, float]]:
        """
        This method
        :param point: d-dimensional Sequence containing the coordinates of the grid point
        :param gridPointCoordsAsStripes:
        :return: d-dimenisional Sequence of 2-dimensional tuples containing the start and end of the function domain in each dimension
        """
        # check if the coordinate is on the boundary and if we have points on the boundary
        boundary_check = lambda x: self.grid.boundary if (x == 0 or x == 1.0) else True

        # create a tuple for each point whose elements are the coordinates that are within the domain
        neighbor_tuple = lambda n: tuple((n[d] for d in range(0, self.dim) if
                                          n[d] >= point_domain[d][0] and n[d] <= point_domain[d][1] and boundary_check(
                                              n[d])))
        if self.debug:
            all_points = list(get_cross_product(gridPointCoordsAsStripes))
            point_domain = self.get_hat_domain(point, gridPointCoordsAsStripes)
            # pick only tuples where both coordinates are within the domain
            neighbors = [neighbor_tuple(p) for p in all_points if len(neighbor_tuple(p)) == self.dim]
            return neighbors
        else:
            all_points = get_cross_product(gridPointCoordsAsStripes)
            point_domain = self.get_hat_domain(point, gridPointCoordsAsStripes)
            # pick only tuples where both coordinates are within the domain
            neighbors = (neighbor_tuple(p) for p in all_points if len(neighbor_tuple(p)) == self.dim)
            return neighbors

    def get_hat_domain(self, point: Sequence[float], gridPointCoordsAsStripes: Sequence[Sequence[float]]):
        """
        This method
        :param point: d-dimensional tuple containing the indices of the point
        :param xxx: Sequence of length d with the maximum level for each dimension
        :return: d-dimenisional Sequence of 2-dimensional tuples containing the start and end of the function domain in each dimension
        """
        # go through stripes and collect 2 coordinates with lowest distance to the point for each dimension
        domain = []
        if self.debug:
            for d in range(0, self.dim):
                upper = [coord for coord in gridPointCoordsAsStripes[d] if coord > point[d]]
                lower = [coord for coord in gridPointCoordsAsStripes[d] if coord < point[d]]
                element = (0 if not lower else max(lower), 1.0 if not upper else min(upper))
                domain.append(element)
        else:
            for d in range(0, self.dim):
                #upper = (coord for coord in gridPointCoordsAsStripes[d] if coord > point[d])
                #lower = (coord for coord in gridPointCoordsAsStripes[d] if coord < point[d])
                element = (max((coord for coord in gridPointCoordsAsStripes[d] if coord < point[d]), default=0.0),
                           min((coord for coord in gridPointCoordsAsStripes[d] if coord > point[d]), default=1.0))
                domain.append(element)
        return domain

    def get_domain_overlap_width(self, point_i: Sequence[float], domain_i: Sequence[Tuple[float, float]],
                                 point_j: Sequence[float], domain_j: Sequence[Tuple[float, float]]) \
            -> Tuple[Sequence[float], Sequence[float]]:
        """
        This method calculates the width of the overlap between the domains of points i and j in each dimension.
        :param point_i:  d-dimensional sequence of coordinates
        :param domain_i: d=dimensional sequence of 2-element tuples with the start and end value of the domain 
        :param point_j:  d-dimensional sequence of coordinates
        :param domain_j: d=dimensional sequence of 2-element tuples with the start and end value of the domain 
        :return: d-dimensional sequence of that describe the width of the overlap between domains and the distance of the points
        """
        # sanity check
        assert len(point_i) == len(point_j) == len(domain_i) == len(domain_j)
        # check adjacency
        if all((domain_i[d][0] <= point_j[d] and domain_i[d][1] >= point_j[d] for d in range(self.dim))):
            widths = []
            distances = []
            for d in range(0, len(point_i)):
                lower = max(domain_i[d][0], domain_j[d][0])
                upper = min(domain_i[d][1], domain_j[d][1])
                widths.append(abs(upper - lower))
                distances.append(abs(point_i[d] - point_j[d]))
            widths.sort()
            distances.sort()
            return (widths, distances)
        else:
            return ([0 for d in range(0, len(point_i))], [0 for d in range(0, len(point_i))])

    def build_R_matrix_dimension_wise(self, gridPointCoordsAsStripes: Sequence[Sequence[float]],
                                      grid_point_levels: Sequence[Sequence[int]]) \
            -> Sequence[Sequence[float]]:
        """
        This method constructs the R matrix for the component grid specified by the levelvector ((R + λ*I) = B) for
        non-equidistant grids (usually used adaptive schemes)
        :param gridPointCoordsAsStripes: d-dimensional sequence of coordinate lists. the lists include coordinates at
                                         the domain boundary, even if there are no boundary points.
        :param grid_point_levels: d-dimensional sequence of integer lists
        :return: R matrix of the component grid specified by the levelvector
        """
        test = list(get_cross_product(gridPointCoordsAsStripes))
        if not self.grid.boundary:
            points = [x for x in test if 0.0 not in x and 1.0 not in x]
        else:
            points = test

        grid_size = len(points)
        R = np.zeros((grid_size, grid_size))

        if self.print_output:
            print("Point list: ", points)
            print("Point levels: ", grid_point_levels)
        if self.reuse_old_values and not self.masslumping:
            # get overlap of domains between points in each dimension in sequence; sort sequence;
            # the string of the sequence
            for i in range(0, len(points)):
                for j in range(i, len(points)):
                    overlap = self.get_domain_overlap_width(points[i], self.get_hat_domain(points[i], gridPointCoordsAsStripes),
                                                            points[j], self.get_hat_domain(points[j], gridPointCoordsAsStripes))
                    #overlap.sort()
                    if str(overlap) in self.old_R:
                        res = self.old_R[str(overlap)]
                    else:
                        if self.numeric_calculation:
                            res = self.calculate_L2_scalarproduct(points[i], self.get_hat_domain(points[i], gridPointCoordsAsStripes),
                                                                  points[j], self.get_hat_domain(points[j], gridPointCoordsAsStripes))[0]
                        else:
                            res = self.calculate_R_value_analytically(points[i], self.get_hat_domain(points[i], gridPointCoordsAsStripes),
                                                                      points[j], self.get_hat_domain(points[j], gridPointCoordsAsStripes))

                        res = res
                        self.old_R[str(overlap)] = res
                    R[i][j] = res
                    R[j][i] = res
        elif not self.reuse_old_values and not self.masslumping:
            # calculate the R matrix elements using the inner product of the hat functions centered at the points i and j
            for i in range(0, len(points)):
                for j in range(i, len(points)):
                    if self.numeric_calculation:
                        res = self.calculate_L2_scalarproduct(points[i], self.get_hat_domain(points[i], gridPointCoordsAsStripes),
                                                              points[j], self.get_hat_domain(points[j], gridPointCoordsAsStripes))[0]
                    else:
                        res = self.calculate_R_value_analytically(points[i], self.get_hat_domain(points[i], gridPointCoordsAsStripes),
                                                                  points[j], self.get_hat_domain(points[j], gridPointCoordsAsStripes))

                    R[i][j] = res
                    R[j][i] = res
        else:
            #only calculate the diagonal
            for i in range(0, len(points)):
                j = i

                if self.numeric_calculation:
                    res = self.calculate_L2_scalarproduct(points[i], self.get_hat_domain(points[i], gridPointCoordsAsStripes),
                                                          points[j], self.get_hat_domain(points[j], gridPointCoordsAsStripes))[0]
                else:
                    res = self.calculate_R_value_analytically(points[i], self.get_hat_domain(points[i], gridPointCoordsAsStripes),
                                                              points[j], self.get_hat_domain(points[j], gridPointCoordsAsStripes))

                R[i][j] = res
                R[j][i] = res

        return R

    def calculate_B_dimension_wise(self, data: Sequence[Sequence[float]], gridPointCoordsAsStripes: Sequence[Sequence[float]], grid_point_levels: Sequence[Sequence[int]]) \
            -> Sequence[float]:
        """
        This method calculates the B vector for the component grid and the data set of the linear system ((R + λ*I) = B)
        :param data: dataset specified for the operation
        :param levelvec: Levelvector of the component grid
        :return: b vector of the component grid
        """
        get_point_list = lambda x: list(get_cross_product(x))
        if not self.grid.boundary:
            point_list = [x for x in get_point_list(gridPointCoordsAsStripes) if 0.0 not in x and 1.0 not in x]
        else:
            point_list = get_point_list(gridPointCoordsAsStripes)

        M = len(data)
        N = len(point_list)
        b = np.zeros(N)

        if self.reuse_old_values:
            old_b_key = self.find_closest_old_B(gridPointCoordsAsStripes)

        if self.reuse_old_values and old_b_key is not None:
            # copy the old values
            old_b = self.old_B[old_b_key]
            old_point_list = [x for x in get_point_list(self.old_grid_coord[old_b_key]) if 0.0 not in x and 1.0 not in x]

            point_domains = [self.get_hat_domain(p, gridPointCoordsAsStripes) for p in point_list]
            old_point_domains = [self.get_hat_domain(p, self.old_grid_coord[old_b_key]) for p in old_point_list]
            domain_match = []
            for i in range(len(point_domains)):
                a = [sum([point_domains[i][d][0] == old[d][0] and point_domains[i][d][1] == old[d][1] for d in range(self.dim)]) == self.dim for old in old_point_domains]
                if True in a:
                    domain_match.append(a.index(True))
                else:
                    domain_match.append(-1)
            for p in range(len(point_list)):
                if point_list[p] in old_point_list and point_list[p] and domain_match[p] != -1:
                    #b[p] = old_b[old_point_list.index(point_list[p])]
                    b[p] = old_b[domain_match[p]]

            # calculate all b points that haven't been copied over (the new points)
            for i in range(len(b)):
                if b[i] == 0:
                    # get the data within the domain of the point
                    domain = self.get_hat_domain(point_list[i], gridPointCoordsAsStripes)
                    data_indices_in_domain = self.find_data_in_domain(domain)
                    # go through all the data points in the intersection set
                    for x in data_indices_in_domain:
                        hat = point_list[i]
                        sign = 1.0
                        if self.classes is not None:
                            sign = -1.0 if self.classes[x] < 1 else 1.0
                        b[i] += (self.hat_function_non_symmetric(hat, domain, data[x]) * sign)
                    b[i] *= (1 / M)
        else:
            for i in range(M):
                hats = self.get_neighbors(data[i], gridPointCoordsAsStripes)
                sign = 1.0
                if self.classes is not None:
                    sign = -1.0 if self.classes[i] < 1 else 1.0
                if self.debug:
                    for j in range(len(hats)):
                        b[point_list.index(hats[j])] += \
                            (self.hat_function_non_symmetric(hats[j], self.get_hat_domain(hats[j], gridPointCoordsAsStripes), data[i]) * sign)
                else:
                    for h in hats:
                        b[point_list.index(h)] += \
                            (self.hat_function_non_symmetric(h, self.get_hat_domain(h, gridPointCoordsAsStripes), data[i]) * sign)
            b *= (1 / M)

        if self.print_output:
            print("B vector: ", b)
        max_levels = [max(x) for x in grid_point_levels]
        self.max_levels = [max(max_levels[d]+1, self.max_levels[d]) for d in range(self.dim)]
        self.new_B[str(max_levels)] = np.array(b)  # copy the values, not the reference
        self.new_grid_coord[str(max_levels)] = [list(g) for g in gridPointCoordsAsStripes]  # copy the values, not the reference
        return b

    def solve_density_estimation_dimension_wise(self, gridPointCoordsAsStripes: Sequence[Sequence[float]], grid_point_levels: Sequence[Sequence[int]], component_grid: ComponentGridInfo) \
            -> Sequence[float]:
        """
        Calculates the surpluses of the component grid for the specified dataset
        :param levelvec: Levelvector of the component grid
        :return: Surpluses of the component grid for the specified dataset
        """
        #numGridCoord = sum((len(l) for l in gridPointCoordsAsStripes))
        r0 = time.time_ns()
        R = self.build_R_matrix_dimension_wise(gridPointCoordsAsStripes, grid_point_levels)
        r1 = time.time_ns()
        print('OP: build_R_matrix_dimension_wise time taken: ', (r1 - r0) / 1000000)
        b0 = time.time_ns()
        b = self.calculate_B_dimension_wise(self.data, gridPointCoordsAsStripes, grid_point_levels)
        b1 = time.time_ns()
        print('OP: calculate_B_dimension_wise time taken: ', (b1 - b0) / 1000000)
        cg0 = time.time_ns()
        alphas, info = cg(R, b)
        cg1 = time.time_ns()
        print('OP: conjugate_gradient time taken: ', (cg1 - cg0) / 1000000)
        if self.print_output:
            print("Alphas: ", component_grid.levelvector, alphas)
            print("-" * 100)
        return alphas

    def calculate_L2_scalarproduct(self, point_i: Tuple[int, ...], domain_i: Sequence[Tuple[int, int]],
                                   point_j: Tuple[int, ...], domain_j: Sequence[Tuple[int, int]]) \
            -> Tuple[float, float]:
        """
        This method calculates the L2-scalarproduct of the two hat functions
        :param ivec: Index of the first hat function
        :param jvec: Index of the second hat function
        :param lvec: Levelvector of the component grid
        :return: L2-scalarproduct of the two hat functions plus the error of the calculation
        """
        if not (len(point_i) == len(point_j) == len(domain_i) == len(domain_j)):
            print('error in calculate_L2_scalarproduct: dimensionality of the points i,j or their domains differ')
        # check adjacency
        if all((domain_i[d][0] <= point_j[d] and domain_i[d][1] >= point_j[d] for d in range(self.dim))):
            f = lambda *x: (self.hat_function_non_symmetric(point_i, domain_i, [*x]) * self.hat_function_non_symmetric(point_j, domain_j, [*x]))
            start = [min(domain_i[d][0], domain_j[d][0]) for d in range(self.dim)]
            end = [max(domain_i[d][1], domain_j[d][1]) for d in range(self.dim)]
            if self.print_output:
                print("-" * 100)
                print("Calculating")
                print("Gridpoints: ", point_i, point_j)
                print("Domain: ", start, end)
            return nquad(f, [[start[d], end[d]] for d in range(self.dim)],
                         opts={"epsabs": 10 ** (-15), "epsrel": 1 ** (-15)})
        else:
            return (0, 0)

    def calculate_R_value_analytically(self, point_i, domain_i, point_j, domain_j):
        # check adjacency
        if not all((domain_i[d][0] <= point_j[d] and domain_i[d][1] >= point_j[d] for d in range(self.dim))):
            return 0.0
        res = 1.0
        for d in range(0, len(point_i)):
            if point_i[d] != point_j[d]:
                m = 1.0 / abs(point_i[d] - point_j[d])  # slope
                # f_2(x) = 1 - slope * (x - min(point_i[d], point_j[d])) = c - slope * x
                a = min(point_i[d], point_j[d])  # lower end of integral
                b = max(point_i[d], point_j[d])  # upper end of integral

                # calc integral of: int (1 - m*(q - x)) * (1 - m*(x - p)) dx
                integral_calc = lambda x, m, p, q: 0.5*(m**2)*(x**2)*(p + q) - (1/3)*(m**2)*(x**3) - x*(m*p + 1)*(m*q - 1)
                #integral_calc_alt = lambda x, m, p, q: x - m*q*x + m*p*x + ((m**2)*q*(x**2))/2 - (m**2)*q*p*x - ((m**2)*(x**3))/3 + ((m**2)*(x**2)*p)/2

                integral = integral_calc(b, m, a, b) - integral_calc(a, m, a, b)

                res *= integral
            else:
                m1 = 1.0 / abs(point_i[d] - domain_i[d][0])  # left slope
                m2 = 1.0 / abs(domain_j[d][1] - point_j[d])  # right slope

                a = domain_i[d][0]  # lower end of first integral
                p = point_i[d] # upper end of first integral, lower end of second integral
                c = domain_i[d][1]  # upper end of second integral

                # calc integral of: int (1 - m*(p - x)) * (1 - m*(p - x)) dx
                integral_1 = lambda x, m, p: -((m * (p - x) - 1)**3 / (3 * m))
                #integral_1_alt = lambda x, m, p: x - 2*m*p*x + m*x**2 + (m**2)*(p**2)*x - (m**2)*p*(x**2) + ((m**2)*(x**3))/3

                # calc integral of: int (1 - m*(x - p)) * (1 - m*(x - p)) dx
                integral_2 = lambda x, m, p: -((m * (p - x) + 1)**3 / (3 * m))
                #integral_2_alt = lambda x, m, p: x + 2*m*p*x - m*x**2 + (m**2)*(p**2)*x - (m**2)*p*(x**2) + ((m**2)*(x**3))/3

                integral = (integral_1(p, m1, p) - integral_1(a, m1, p)) + (integral_2(c, m2, p) - integral_2(p, m2, p))

                res *= integral
        return res

    def hat_function_non_symmetric(self, point: Sequence[float], domain: Sequence[float], x: Sequence[float]) \
            -> float:
        """
        This method calculates the hat function value of the given coordinates with the given parameters
        :param : d-dimensional center point of the hat function
        :param : d-dimensional list of 2-dimensional tuples that describe the start and end values of the domain of the hat function
        :param : d-dimensional coordinates whose function value are to be calculated
        :return: value of the function at the coordinates given by x
        """
        assert len(point) == len(x) == len(domain)   # sanity check
        result = 1.0
        if not self.grid.modified_basis:
            for dim in range(len(x)):
                if x[dim] >= point[dim]:
                    # result is linear interpolation between middle and domain end
                    result *= max(0.0, 1.0 - (1.0 / (domain[dim][1] - point[dim])) * (x[dim] - point[dim]))
                elif x[dim] < point[dim]:
                    result *= max(0.0, 1.0 - (1.0 / (point[dim] - domain[dim][0])) * (point[dim] - x[dim]))
            return result
        else:
            for dim in range(len(x)):
                # if the domain reaches the boundary, we extrapolate with the same slope that's to the neighboring point
                boundary_check = lambda x: x == 0.0 or x == 1.0
                if x[dim] >= point[dim]:
                    # result is linear interpolation between middle and domain end
                    if boundary_check(domain[dim][1]):
                        result *= (1.0 / (domain[dim][1] - point[dim])) * (x[dim] - point[dim])
                    else:
                        result *= max(0.0, 1.0 - (1.0 / (domain[dim][1] - point[dim])) * (x[dim] - point[dim]))
                elif x[dim] < point[dim]:
                    if boundary_check(domain[dim][0]):
                        result *= (1.0 / (point[dim] - domain[dim][0])) * (point[dim] - x[dim])
                    else:
                        result *= max(0.0, 1.0 - (1.0 / (point[dim] - domain[dim][0])) * (point[dim] - x[dim]))
            return result

    def find_closest_old_B(self, gridPointCoordinatesAsStripes):
        """
        This method looks for the closest match of old B vectors for the current grid and returns it, along with
        a range of indices that need to be changed/updated
        Parameters
        ----------
        gridPointCoordinatesAsStripes

        Returns
        -------

        """

        if len(self.old_B) == 0:
            return None

        # get the ranges for all new coordinates
        new_coordinate_sets = []
        key_list = []
        for key in self.old_B.keys():
            old_coordinates = self.old_grid_coord[key]
            key_list.append(key)
            new_coordinates_indices = []
            for d in range(self.dim):
                new_coordinates_indices.append([])
                for i in range(len(gridPointCoordinatesAsStripes[d])):
                    if gridPointCoordinatesAsStripes[d][i] not in old_coordinates[d]:
                        new_coordinates_indices[d].append(gridPointCoordinatesAsStripes[d].index(gridPointCoordinatesAsStripes[d][i]))
            new_coordinate_sets.append(new_coordinates_indices)

        # get the coordinate list of
        differences = [sum([len(i) for i in s]) for s in new_coordinate_sets]
        new_coordinates_indices = new_coordinate_sets[differences.index(min(differences))]
        closest_match_key = key_list[differences.index(min(differences))]

        # get the new points as list
        new_coordinates = [[gridPointCoordinatesAsStripes[d][x] for x in new_coordinates_indices[d]] for d in range(self.dim)]
        new_points = []
        for d in range(self.dim):
            if len(new_coordinates[d]) > 0:
                coords = [gridPointCoordinatesAsStripes[x] if x == d else new_coordinates[x] for x in range(len(gridPointCoordinatesAsStripes))]
                points = list(get_cross_product(coords))
                new_points.append(points)


        return closest_match_key


    def find_enclosing_bin(self, domain: Tuple[float, float], dim: int):
        """
        This function looks for indices in the bins for the given dimension that are the closest to the given domain.
        (Only relevant for reuse of old B vectors)
        domain: 2-dimensional Sequence; Domain for which the closest indices should be found in the sorted data
        dim:    The dimension for which the enclosing bin should be found
        :return: 2-dimensional Sequence; Closest indices to the given domain, in the given dimension within the sorted data
        """
        enclosing_bin = [0, len(self.sorted_data[dim] - 1)]
        # check for available bins
        if domain[0] == 0.0 and domain[1] == 1.0:
            return enclosing_bin
        if dim in self.data_bins:
            bins = self.data_bins[dim]
        else:
            return enclosing_bin
        if len(bins) == 0:
            return enclosing_bin
        # check if we have an enclosing bin, so we don't have to search the entire data set
        extract_list = lambda x: list(map(float, x[1:-1].split(',')))
        enclosing_bin_keys = [extract_list(key_string) for key_string in bins.keys()]

        lower_end = min([x for x in enclosing_bin_keys if x[0] <= domain[0]], key=lambda t: abs(t[0] - domain[0]))[0]
        higher_end = min([x for x in enclosing_bin_keys if x[1] >= domain[1]], key=lambda t: abs(t[1] - domain[1]))[1]

        enclosing_bin = [lower_end, higher_end]
        return enclosing_bin

    def find_data_in_domain(self, domain: Sequence[Tuple[float, float]]):
        """
        This method finds all data points
        domain:
        :return:
        """
        data_ranges = []

        for d in range(self.dim):
            data_ranges.append([])
            # check if we have a data bin for the domain, otherwise create the bin
            key = str([domain[d][0], domain[d][1]])
            if key in self.data_bins[d]:
                data_ranges[d] = [self.data_bins[d][key][0], self.data_bins[d][key][1]]
            else:
                enclosing_bin = self.find_enclosing_bin(domain[d], d)
                lower = len(self.sorted_data[d])
                upper = 0
                # find the lowest min and highest max within the domain
                for i in range(enclosing_bin[0], min(enclosing_bin[1]+1, len(self.sorted_data[d]))):
                    if self.data[self.sorted_data[d][i]][d] >= domain[d][0] and i < lower:
                        lower = i
                    if self.data[self.sorted_data[d][i]][d] <= domain[d][1] and i > upper:
                        upper = i
                data_ranges[d] = [max(lower-1, 0), min(upper+1, len(self.sorted_data[d])-1)]
        # save the data ranges as bins
        for d in range(self.dim):
            self.data_bins[d][str([domain[d][0], domain[d][1]])] = data_ranges[d]

        domain_data = self.sorted_data[0][data_ranges[0][0]:data_ranges[0][1]]
        for d in range(self.dim):
            domain_data = np.intersect1d(domain_data, self.sorted_data[d][data_ranges[d][0]:data_ranges[d][1]])
        return domain_data

    def build_R_matrix(self, levelvec: Sequence[int]) -> Sequence[Sequence[float]]:
        """
        This method constructs the R matrix for the component grid specified by the levelvector ((R + λ*I) = B)
        :param levelvec: Levelvector of the component grid
        :return: R matrix of the component grid specified by the levelvector
        """
        grid_size = self.grid.get_num_points()
        R = np.zeros((grid_size, grid_size))
        dim = len(levelvec)

        #if not self.grid.is_global():
        index_list = self.grid.get_indexlist()
        #else:
        #    index_list = self.get_existing_indices(levelvec)


        diag_val = np.prod([1 / (2 ** (levelvec[k] - 1) * 3) for k in range(dim)])
        R[np.diag_indices_from(R)] += (diag_val + self.lambd)
        if self.print_output:
            print("Indexlist: ", index_list)
            print("Levelvector: ", levelvec)
            print("Diagonal value: ", diag_val)
        if self.masslumping == False:
            for i in range(grid_size - 1):
                for j in range(i + 1, grid_size):
                    res = 1.0

                    for k in range(dim):
                        index_ik = index_list[i][k]
                        index_jk = index_list[j][k]

                        # basis function overlap fully
                        if index_ik == index_jk:
                            res *= 1 / (2 ** (levelvec[k] - 1) * 3)
                        # basis function do not overlap
                        elif max((index_ik - 1) * 2 ** (levelvec[k] - 1), (index_jk - 1) * 2 ** (levelvec[k] - 1)) \
                                >= min((index_ik + 1) * 2 ** (levelvec[k] - 1), (index_jk + 1) * 2 ** (levelvec[k] - 1)):
                            res = 0
                            break
                        # basis functions overlap partly
                        else:
                            res *= 1 / (2 ** (levelvec[k] - 1) * 12)

                    if res == 0:
                        if self.print_output:
                            print("-" * 100)
                            print("Skipping calculation")
                            print("Gridpoints: ", index_list[i], index_list[j])
                    else:
                        if self.print_output:
                            print("-" * 100)
                            print("Calculating")
                            print("Gridpoints: ", index_list[i], index_list[j])
                            print("Result: ", res)
                        R[i, j] = res
                        R[j, i] = res
        return R

    def solve_density_estimation(self, levelvec: Sequence[int]) -> Sequence[float]:
        """
        Calculates the surpluses of the component grid for the specified dataset
        :param levelvec: Levelvector of the component grid
        :return: Surpluses of the component grid for the specified dataset
        """
        R = self.build_R_matrix(levelvec)
        b = self.calculate_B(self.data, levelvec)
        alphas, info = cg(R, b)
        if self.print_output:
            print("Alphas: ", levelvec, alphas)
            print("-" * 100)
        return alphas

    def calculate_B(self, data: Sequence[Sequence[float]], levelvec: Sequence[int]) -> Sequence[float]:
        """
        This method calculates the B vector for the component grid and the data set of the linear system ((R + λ*I) = B)
        :param data: dataset specified for the operation
        :param levelvec: Levelvector of the component grid
        :return: b vector of the component grid
        """
        M = len(data)
        N = self.grid.get_num_points()
        b = np.zeros(N)

        index_list = self.grid.get_indexlist()

        if self.reuse_old_values:
            get_point_list = lambda x: list(get_cross_product(x))
            gridPointCoordsAsStripes = [[(1 / (2**levelvec[d])) * (i+1) for i in range((2**levelvec[d])-1)] for d in range(self.dim)]

            if not self.grid.boundary:
                point_list = [x for x in get_point_list(gridPointCoordsAsStripes) if 0.0 not in x and 1.0 not in x]
            else:
                point_list = get_point_list(gridPointCoordsAsStripes)

            old_b_key = self.find_closest_old_B(gridPointCoordsAsStripes)

        if self.reuse_old_values and old_b_key is not None:
            # copy the old values
            old_b = self.old_B[old_b_key]
            old_point_list = [x for x in get_point_list(self.old_grid_coord[old_b_key]) if 0.0 not in x and 1.0 not in x]

            # new_points = [p for p in point_list if p not in old_point_list]
            # new_point_neighbors = [self.get_neighbors(p, gridPointCoordsAsStripes) for p in new_points]
            # affected_points = new_points
            # for l in new_point_neighbors:
            #     affected_points += l

            point_domains = [self.get_hat_domain(p, gridPointCoordsAsStripes) for p in point_list]
            old_point_domains = [self.get_hat_domain(p, self.old_grid_coord[old_b_key]) for p in old_point_list]
            domain_match = []
            for i in range(len(point_domains)):
                a = [sum([point_domains[i][d][0] == old[d][0] and point_domains[i][d][1] == old[d][1] for d in range(self.dim)]) == self.dim for old in old_point_domains]
                if True in a:
                    domain_match.append(a.index(True))
                else:
                    domain_match.append(-1)
            for p in range(len(point_list)):
                if point_list[p] in old_point_list and point_list[p] and domain_match[p] != -1:
                    #b[p] = old_b[old_point_list.index(point_list[p])]
                    b[p] = old_b[domain_match[p]]

            # calculate all b points that haven't been copied over (the new points)
            for i in range(len(b)):
                if b[i] == 0:
                    # get the data within the domain of the point
                    #print('recalc b i', i)
                    domain = self.get_hat_domain(point_list[i], gridPointCoordsAsStripes)
                    data_indices_in_domain = self.find_data_in_domain(domain)
                    # go through all the data points in the intersection set
                    for x in data_indices_in_domain:
                        hat = point_list[i]
                        sign = 1.0
                        if self.classes is not None:
                            sign = -1.0 if self.classes[x] < 1 else 1.0
                        b[i] += (self.hat_function_non_symmetric(hat, domain, data[x]) * sign)
                    b[i] *= (1 / M)
        else:
            for i in range(M):
                hats = self.get_hats_in_support(levelvec, data[i])
                for j in range(len(hats)):
                    sign = 1.0
                    if self.classes is not None:
                        sign = -1.0 if self.classes[i] < 1 else 1.0
                    b[index_list.index(hats[j])] += (self.hat_function(hats[j], levelvec, data[i]) * sign)
            b *= (1 / M)
            if self.print_output:
                print("B vector: ", b)

        return b

    def hat_function(self, ivec: Sequence[int], lvec: Sequence[int], x: Sequence[float]) -> float:
        """
        This method calculates the value of the hat function at the point x
        :param ivec: Index of the hat function
        :param lvec: Levelvector of the component grid
        :param x: datapoint
        :return: Value of the hat function at x
        """
        dim = len(lvec)
        result = 1.0
        for d in range(dim):
            result *= max((1 - abs(2 ** lvec[d] * x[d] - ivec[d])), 0)
        return result

    def weighted_basis_function(self, levelvec: Sequence[int], alphas: Sequence[float], x: Sequence[float]) -> float:
        """
        This method calculates the sum of basis functions of the component grid,
        in whose support the data point x lies, weighted by the specific surpluses
        :param levelvec: Levelvector of the compoenent grid
        :param alphas: the calculated surpluses of the component grid
        :param x: datapoint
        :return: Sum of basis functions of the component grid in whose support the data point x lies, weighted by the surpluses
        """
        index_list = self.grid.get_indexlist()
        hats_in_support = self.get_hats_in_support(levelvec, x)
        sum = 0
        for i, index in enumerate(hats_in_support):
            sum += self.hat_function(index, levelvec, x) * alphas[index_list.index(index)]
        return sum

    def plot_dataset(self, filename: str = None):
        """
        This method plots the data set specified for this operation
        :param filename: If set the plot will be saved to the specified filename
        :return: Matplotlib figure
        """
        if self.initialized == False:
            self.initialize()
        fontsize = 30
        plt.rcParams.update({'font.size': fontsize})
        fig = plt.figure(figsize=(10, 10))
        if self.dim == 2:
            ax = fig.add_subplot(1, 1, 1)
            x, y = zip(*self.data[:, :self.dim])
            ax.scatter(x, y, s=125)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title("M = %d" % len(self.data[:, :self.dim]))

        elif self.dim == 3:
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            x, y, z = zip(*self.data[:, :self.dim])
            ax.scatter(x, y, z, s=125)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_title("#points = %d" % len(self.data[:, :self.dim]))

        else:
            print("Cannot print data of dimension > 2")

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
        #plt.show()
        # reset fontsize to default so it does not affect other figures
        plt.rcParams.update({'font.size': plt.rcParamsDefault.get('font.size')})
        return fig

    def plot_component_grid(self, combiObject: "StandardCombi", component_grid: ComponentGridInfo, grid: Axes3D, pointsPerDim: int = 100) -> None:
        """
        This method plots the contour plot of the component grid specified by the ComponentGridInfo.
        This method is used by print_resulting_combi_scheme in StandardCombi
        :param component_grid:  ComponentGridInfo of the specified component grid.
        :param grid: Axes3D of the
        :param levels: the amount of different levels for the contourf plot
        :param pointsPerDim: amount of points to be plotted in each dimension
        :return: None
        """
        X = np.linspace(0.0, 1.0, pointsPerDim)
        Y = np.linspace(0.0, 1.0, pointsPerDim)
        X, Y = np.meshgrid(X, Y)
        Z = combiObject.interpolate_points(list(map(lambda x, y: (x, y), X.flatten(), Y.flatten())), component_grid)
        Z = Z.reshape((100, 100))
        t = cm.coolwarm
        tt = colors.PowerNorm(gamma=0.95, vmin=self.extrema[0], vmax=self.extrema[1])
        grid.imshow(Z, extent=[0.0, 1.0, 0.0, 1.0], origin='lower', cmap=cm.coolwarm, norm=colors.PowerNorm(gamma=0.95, vmin=self.extrema[0], vmax=self.extrema[1]))

    def print_evaluation_output(self, refinement):
        combi_surpluses = self.surpluses
        if len(combi_surpluses) == 1:
            combi_surpluses = combi_surpluses[0]
        #print("combisurpluses:", combi_surpluses)

    def get_global_error_estimate(self, refinement_container, norm):
        if self.reference_solution is None:
            return None
        elif LA.norm(self.reference_solution) == 0.0:
            return LA.norm(abs(self.surpluses), norm)
        else:
            return LA.norm(abs((self.reference_solution - self.surpluses) / self.reference_solution), norm)


from scipy.interpolate import interpn


class Integration(AreaOperation):
    def __init__(self, f: Function, grid: Grid, dim: int, reference_solution: Sequence[float] = None):
        self.f = f
        self.f_actual = None
        self.grid = grid
        self.reference_solution = reference_solution
        self.dim = dim
        self.dict_integral = {}
        self.dict_points = {}
        self.integral = np.zeros(f.output_length())

    def get_distinct_points(self):
        return self.f.get_f_dict_size()

    def get_point_values_component_grid(self, points, component_grid) -> Sequence[Sequence[float]]:
        """This method returns the values in the component grid at the given points.

        :param points: Points where we want to evaluate the componenet grid (should coincide with grid points)
        :param component_grid: Component grid which we want to evaluate.
        :return: Values at points (same order).
        """
        return np.asarray([self.f(p) for p in points])

    def process_removed_objects(self, removed_objects: List[RefinementObject]) -> None:
        for removed_object in removed_objects:
            self.integral -= removed_object.value

    def get_component_grid_values(self, component_grid, mesh_points_grid):
        mesh_points = get_cross_product(mesh_points_grid)
        function_value_dim = self.f.output_length()
        # calculate function values at mesh points and transform  correct data structure for scipy
        values = np.array(
            [self.f(p) if self.grid.point_not_zero(p) else np.zeros(function_value_dim) for p in mesh_points])
        return values

    def get_mesh_values(self, mesh_points_grid):
        mesh_points = get_cross_product(mesh_points_grid)
        function_value_dim = self.f.output_length()
        # calculate function values at mesh points and transform  correct data structure for scipy
        values = np.array(
            [self.f(p) if self.grid.point_not_zero(p) else np.zeros(function_value_dim) for p in mesh_points])
        return values

    def get_result(self):
        return self.integral

    def point_output_length(self):
        return self.f.output_length()

    def initialize(self):
        self.f.reset_dictionary()
        self.integral = np.zeros(self.f.output_length())

    def eval_analytic(self, coordinate: Tuple[float, ...]) -> Sequence[float]:
        return self.f.eval(coordinate)

    def add_value(self, combined_solution: Sequence[float], new_solution: Sequence[float], component_grid_info: ComponentGridInfo):
        return combined_solution + component_grid_info.coefficient * new_solution

    def evaluate_area(self, area, levelvector, componentgrid_info, refinement_container, additional_info):
        partial_integral = self.grid.integrate(self.f, levelvector, area.start, area.end)
        if area.value is None:
            area.value = partial_integral * componentgrid_info.coefficient
        else:
            area.value += partial_integral * componentgrid_info.coefficient
        evaluations = np.prod(self.grid.levelToNumPoints(levelvector))
        if refinement_container is not None:
            refinement_container.value += partial_integral * componentgrid_info.coefficient
        self.integral += partial_integral * componentgrid_info.coefficient
        return evaluations

    def evaluate_levelvec(self, component_grid: ComponentGridInfo):
        levelvector = component_grid.levelvector
        partial_integral = self.grid.integrate(self.f, levelvector, self.grid.a, self.grid.b)
        self.integral += partial_integral * component_grid.coefficient

    def evaluate_area_for_error_estimates(self, area, levelvector, componentgrid_info, refinement_container, additional_info):
        if additional_info.error_name == "extend_parent":
            assert additional_info.filter_area is None
            extend_parent_new = self.grid.integrate(self.f, levelvector, area.start, area.end)
            if area.parent_info.extend_parent_integral is None:
                area.parent_info.extend_parent_integral = extend_parent_new * componentgrid_info.coefficient
            else:
                area.parent_info.extend_parent_integral += extend_parent_new * componentgrid_info.coefficient
            return np.prod(self.grid.levelToNumPoints(levelvector))
        elif additional_info.error_name == "extend_error_correction":
            assert additional_info.filter_area is None
            if area.switch_to_parent_estimation:
                extend_parent_new = self.grid.integrate(self.f, levelvector, area.start, area.end)
                if area.parent_info.extend_error_correction is None:
                    area.parent_info.extend_error_correction = np.array(area.value)
                area.parent_info.extend_error_correction -= extend_parent_new * componentgrid_info.coefficient
                return np.prod(self.grid.levelToNumPoints(levelvector))
            else:
                return 0
        elif additional_info.error_name == "split_no_filter":
            assert additional_info.filter_area is None
            split_parent_new = self.grid.integrate(self.f, levelvector, area.start, area.end)
            if additional_info.target_area.parent_info.split_parent_integral is None:
                additional_info.target_area.parent_info.split_parent_integral = split_parent_new * componentgrid_info.coefficient
            else:
                additional_info.target_area.parent_info.split_parent_integral += split_parent_new * componentgrid_info.coefficient
            return np.prod(self.grid.levelToNumPoints(levelvector))
        else:
            assert additional_info.filter_area is not None
            if not additional_info.interpolate:  # use filtering approach
                self.grid.setCurrentArea(area.start, area.end, levelvector)
                points, weights = self.grid.get_points_and_weights()
                integral = 0.0
                num_points = 0
                for i, (p, weight) in enumerate(zip(points, weights)):
                    if additional_info.filter_area.point_in_area(p):
                        integral += self.f(p) * weight * self.get_point_factor(p, additional_info.filter_area, area)
                        num_points += 1
            else:  # use bilinear interpolation to get function values in filter_area
                integral = 0.0
                num_points = 0
                # create grid with boundaries; if 0 boudnary condition we will fill boundary points with 0's
                boundary_save = self.grid.get_boundaries()
                self.grid.set_boundaries([True] * self.dim)
                self.grid.setCurrentArea(area.start, area.end, levelvector)
                self.grid.set_boundaries(boundary_save)

                # get corner grid in scipy data structure
                mesh_points_grid = [self.grid.coordinate_array[d] for d in range(self.dim)]

                # get points of filter area for which we want interpolated values
                self.grid.setCurrentArea(additional_info.filter_area.start, additional_info.filter_area.end, levelvector)
                points, weights = self.grid.get_points_and_weights()

                # bilinear interpolation
                interpolated_values = self.interpolate_points(self.get_component_grid_values(componentgrid_info, mesh_points_grid), mesh_points_grid,
                                                              points)

                integral += np.inner(interpolated_values.T, weights)

                # calculate all mesh points
                mesh_points = list(
                    zip(*[g.ravel() for g in np.meshgrid(*[mesh_points_grid[d] for d in range(self.dim)])]))

                # count the number of mesh points that fall into the filter area
                for p in mesh_points:
                    if additional_info.filter_area.point_in_area(p) and self.grid.point_not_zero(p):
                        num_points += 1
            if additional_info.error_name == "split_parent":
                child_area = additional_info.filter_area
                if child_area.parent_info.split_parent_integral is None:
                    child_area.parent_info.split_parent_integral = 0.0
                child_area.parent_info.split_parent_integral += integral * componentgrid_info.coefficient
            else:
                if additional_info.error_name == "split_parent2":
                    child_area = additional_info.filter_area
                    if child_area.parent_info.split_parent_integral2 is None:
                        child_area.parent_info.split_parent_integral2 = 0.0
                    child_area.parent_info.split_parent_integral2 += integral * componentgrid_info.coefficient

                else:
                    assert additional_info.error_name == "reference"
            return num_points

    def count_unique_points(self):
        return True

    def area_preprocessing(self, area):
        # area.set_integral(0.0)
        area.set_value(np.zeros(self.f.output_length()))
        # area.evaluations = 0
        # area.levelvec_dict = {}
        # area.error = None

    def get_global_error_estimate(self, refinement_container, norm):
        if self.reference_solution is None:
            return None
        elif LA.norm(self.reference_solution) == 0.0:
            return LA.norm(abs(self.integral), norm)
        else:
            return LA.norm(abs((self.reference_solution - self.integral) / self.reference_solution), norm)

    #    def area_postprocessing(self, area):
    #        area.value = np.array(area.integral)

    def get_point_factor(self, point, area, area_parent):
        factor = 1.0
        for d in range(self.dim):
            if (point[d] == area.start[d] or point[d] == area.end[d]) and not (
                    point[d] == area_parent.start[d] or point[d] == area_parent.end[d]):
                factor /= 2.0
        return factor

    # interpolates the cell at the subcell edge points and evaluates the integral based on the trapezoidal rule
    def compute_subcell_with_interpolation(self, cell, subcell, coefficient, refinement_container):
        start_subcell = subcell.start
        end_subcell = subcell.end
        start_cell = cell.start
        end_cell = cell.end
        subcell_points = list(zip(*[g.ravel() for g in np.meshgrid(*[[start_subcell[d], end_subcell[d]] for d in range(self.dim)])]))
        corner_points_grid = [[start_cell[d], end_cell[d]] for d in range(self.dim)]
        interpolated_values = self.interpolate_points(self.get_mesh_values(corner_points_grid), corner_points_grid, subcell_points)
        width = np.prod(np.array(end_subcell) - np.array(start_subcell))
        factor = 0.5 ** self.dim * width
        integral = 0.0
        for p in interpolated_values:
            integral += p * factor
        subcell.cell_dict[subcell.get_key()].sub_integrals.append((integral, coefficient))
        subcell.value += integral * coefficient
        if refinement_container is not None:
            refinement_container.value += integral * coefficient
        self.integral += integral * coefficient

    def print_evaluation_output(self, refinement):
        combi_integral = self.integral
        if len(combi_integral) == 1:
            combi_integral = combi_integral[0]
        print("combiintegral:", combi_integral)

    def calculate_operation_dimension_wise(self, gridPointCoordsAsStripes, grid_point_levels, component_grid):
        reuse_old_values = False
        if reuse_old_values:
            previous_integral, previous_points = self.get_previous_integral_and_points(component_grid.levelvector)
            integral = np.array(previous_integral)
            previous_points_coarsened = list(previous_points)
            modification_points, modification_points_coarsen = self.get_modification_points(previous_points,
                                                                                            gridPointCoordsAsStripes)
            if modification_points_coarsen is not None:
                for d in range(self.dim):
                    previous_points_coarsened[d] = list(previous_points[d])
                    for mod_point in modification_points_coarsen[d]:
                        for removal_point in mod_point[1]:
                            previous_points_coarsened[d].remove(removal_point)
                integral += self.subtract_contributions(modification_points_coarsen, previous_points_coarsened,
                                                        previous_points)
                integral -= self.get_new_contributions(modification_points_coarsen, previous_points)
            if modification_points is not None:
                # ~ integral -= self.subtract_contributions(modification_points, previous_points_coarsened,
                # ~ gridPointCoordsAsStripes)
                v = self.subtract_contributions(modification_points, previous_points_coarsened,
                                                gridPointCoordsAsStripes)
                assert len(v) == len(integral)
                integral -= v
                integral += self.get_new_contributions(modification_points, gridPointCoordsAsStripes)
        else:
            self.grid_surplusses.set_grid(gridPointCoordsAsStripes, grid_point_levels)
            self.grid.set_grid(gridPointCoordsAsStripes, grid_point_levels)
            integral = self.grid.integrate(self.f, component_grid.levelvector, self.a, self.b)
        self.refinement_container.value += integral * component_grid.coefficient
        self.integral += integral * component_grid.coefficient
        if reuse_old_values:
            self.dict_integral[tuple(component_grid.levelvector)] = np.array(integral)
            self.dict_points[tuple(component_grid.levelvector)] = np.array(gridPointCoordsAsStripes)

    def set_function(self, f=None):
        assert f is None or f == self.f, "Integration and the refinement should use the same function"

    def init_dimension_wise(self, grid, grid_surplusses, refinement_container, lmin, lmax, a, b, version=2):
        self.grid = grid
        self.grid_surplusses = grid_surplusses
        self.refinement_container = refinement_container
        self.version = version
        self.lmin = lmin
        self.lmax = lmax
        self.a = a
        self.b = b

    def initialize_evaluation_dimension_wise(self, refinement_container):
        refinement_container.value = np.zeros(self.f.output_length())
        self.integral = np.zeros(self.f.output_length())

    # This method returns the previous integral approximation + the points contained in this grid for the given
    # component grid identified by the levelvector. In case the component grid is new, we search for a close component
    # grid with levelvector2 <= levelvector and return the respective previous integral and the points of the
    # previous grid.
    def get_previous_integral_and_points(self, levelvector):
        if tuple(levelvector) in self.dict_integral:
            return np.array(self.dict_integral[tuple(levelvector)]), np.array(self.dict_points[tuple(levelvector)])
        else:
            k = 1
            dimensions = []
            for d in range(self.dim):
                if self.lmax[d] - k > 0:
                    dimensions.append(d)
            while k < max(self.lmax):
                reduction_values = list(zip(*[g.ravel() for g in np.meshgrid(
                    *[range(0, min(k, self.lmax[d] - self.lmin[d])) for d in range(self.dim)])]))

                for value in reduction_values:
                    levelvec_temp = np.array(levelvector) - np.array(list(value))
                    if tuple(levelvec_temp) in self.dict_integral:
                        return np.array(self.dict_integral[tuple(levelvec_temp)]), np.array(self.dict_points[tuple(levelvec_temp)])
                k += 1
        assert False

    # This method checks if there are new points in the grid new_points compared to the old grid old_points
    # We then return a suited data structure containing the newly added points and the points that were removed.
    def get_modification_points(self, old_points, new_points):
        found_modification = found_modification2 = False
        # storage for newly added points per dimension
        modification_array_added = [[] for d in range(self.dim)]
        # storage for removed points per dimension
        modification_arra_removed = [[] for d in range(self.dim)]

        for d in range(self.dim):
            # get newly added points for dimension d
            modifications = sorted(list(set(new_points[d]) - set(old_points[d])))
            if len(modifications) != 0:
                found_modification = True
                modification_1D = self.get_modification_objects(modifications, new_points[d])
                modification_array_added[d].extend(list(modification_1D))
            # get removed points for dimension d
            modifications_coarsen = sorted(list(set(old_points[d]) - set(new_points[d])))
            if len(modifications_coarsen) != 0:
                found_modification2 = True
                modification_1D = self.get_modification_objects(modifications_coarsen, old_points[d])
                modification_arra_removed[d].extend(list(modification_1D))
        return modification_array_added if found_modification else None, modification_arra_removed if found_modification2 else None

    # Construct the data structures for the newly added points listed in modifications. The complete grid is given in
    # grid_points.
    def get_modification_objects(self, modifications, grid_points):
        modification_1D = []
        k = 0
        for i in range(len(grid_points)):
            if grid_points[i] == modifications[k]:
                j = 1
                # get consecutive list of points that are newly added
                while k + j < len(modifications) and grid_points[i + j] == modifications[k + j]:
                    j += 1
                # store left and right neighbour in addition to the newly added points list(grid_points[i:i + j])
                modification_1D.append((grid_points[i - 1], list(grid_points[i:i + j]), grid_points[i + j]))
                k += j
                if k == len(modifications):
                    break
        return modification_1D

    # This method calculates the change of the integral contribution of the neighbouring points of newly added points.
    # We assume here a trapezoidal rule. The newly added points are contained in new_points but not in old_points.
    def subtract_contributions(self, modification_points, old_points, new_points):
        # calculate weights of point in new grid
        self.grid_surplusses.set_grid(new_points)
        weights = self.grid_surplusses.weights
        # save weights in dictionary for fast access via coordinate
        dict_weights_fine = [{} for d in range(self.dim)]
        for d in range(self.dim):
            for p, w in zip(self.grid_surplusses.coords[d], weights[d]):
                dict_weights_fine[d][p] = w
        # reset grid to old grid
        self.grid_surplusses.set_grid(old_points)
        # sum up the changes in contributions
        integral = 0.0
        for d in range(self.dim):
            for point in modification_points[d]:
                # calculate the changes in contributions for all points that contain the neighbouring points point[0]
                # and point[2] in dimension d
                points_for_slice = list([point[0], point[2]])
                # remove boundary points if contained if grid has no boundary points
                if not self.grid.boundary:
                    points_for_slice = [p for p in points_for_slice if not (isclose(p, self.a[d]) or isclose(p, self.b[d]))]
                integral += self.calc_slice_through_points(points_for_slice, old_points, d, modification_points, subtract_contribution=True,
                                                           dict=dict_weights_fine)
        return integral

    # This method calculates the new contributions of the points specified in modification_points to the grid new_points
    # The new_points grid contains the newly added points.
    def get_new_contributions(self, modification_points, new_points):
        self.grid_surplusses.set_grid(new_points)
        # sum up all new contributions
        integral = 0.0
        for d in range(self.dim):
            for point in modification_points[d]:
                # calculate the new contribution of the points with the new coordinates points[1] (a list of one or
                # multiple new coordinates) in dimension d
                integral += self.calc_slice_through_points(point[1], new_points, d, modification_points)
        return integral

    # This method computes the integral of the dim-1 dimensional slice through the points_for_slice of dimension d.
    # We also account for the fact that some points might be traversed by multiple of these slice calculations and
    # reduce the factors accordingly. If subtract_contribution is set we calculate the difference of the
    # new contribution from previously existing points to the new points.
    def calc_slice_through_points(self, points_for_slice, grid_points, d, modification_points, subtract_contribution=False, dict=None):
        integral = 0.0
        positions = [list(self.grid_surplusses.coords[d]).index(point) for point in points_for_slice]
        points = list(
            zip(*[g.ravel() for g in np.meshgrid(*[self.grid_surplusses.coords[d2] if d != d2 else points_for_slice for d2 in range(self.dim)])]))
        indices = list(zip(
            *[g.ravel() for g in np.meshgrid(*[range(len(self.grid_surplusses.coords[d2])) if d != d2 else positions for d2 in range(self.dim)])]))
        for i in range(len(points)):
            # index of current point in grid_points grid
            index = indices[i]
            # point coordinates of current point
            current_point = points[i]
            # old weight of current point in coarser grid
            weight = self.grid_surplusses.getWeight(index)
            if subtract_contribution:
                # weight of current point in new finer grid
                weight_fine = 1
                for d in range(self.dim):
                    weight_fine *= dict[d][current_point[d]]
                number_of_dimensions_that_intersect = 0
                # calculate if other slices also contain this point
                for d2 in range(self.dim):
                    for mod_point in modification_points[d2]:
                        if current_point[d2] == mod_point[0] or current_point[d2] == mod_point[2]:
                            number_of_dimensions_that_intersect += 1
                # calculate the weight difference from the old to the new grid
                factor = (weight - weight_fine) / number_of_dimensions_that_intersect
            else:
                number_of_dimensions_that_intersect = 1
                # calculate if other slices also contain this point
                for d2 in range(self.dim):
                    if d2 == d:
                        continue
                    for mod_point in modification_points[d2]:
                        if current_point[d2] in mod_point[1]:
                            number_of_dimensions_that_intersect += 1
                # calculate the new weight contribution of newly added point
                factor = weight / number_of_dimensions_that_intersect
            assert (factor >= 0)
            integral += self.f(current_point) * factor
        return integral


class Interpolation(Integration):
    # interpolates mesh_points_grid at the given  evaluation_points using bilinear interpolation
    @staticmethod
    def interpolate_points(values: Sequence[Sequence[float]], dim: int, grid: Grid, mesh_points_grid: Sequence[Sequence[float]],
                           evaluation_points: Sequence[Tuple[float, ...]]):
        # constructing all points from mesh definition
        function_value_dim = len(values[0])
        interpolated_values_array = []
        for d in range(function_value_dim):
            values_1D = np.asarray([value[d] for value in values])

            values_1D = values_1D.reshape(*[len(mesh_points_grid[d]) for d in (range(dim))])

            # interpolate evaluation points from mesh points with bilinear interpolation
            interpolated_values = interpn(mesh_points_grid, values_1D, evaluation_points, method='linear')

            interpolated_values = np.asarray([[value] for value in interpolated_values])
            interpolated_values_array.append(interpolated_values)
        return np.hstack(interpolated_values_array)


import chaospy as cp
import scipy.stats as sps
from Function import *
from StandardCombi import *  # For reference solution calculation


class UncertaintyQuantification(Integration):
    # The constructor resembles Integration's constructor;
    # it has an additional parameter:
    # distributions can be a list, tuple or string
    def __init__(self, f, distributions, a: Sequence[float], b: Sequence[float],
                 dim: int = None, grid=None, reference_solution=None):
        dim = dim or len(a)
        super().__init__(f, grid, dim, reference_solution)
        self.f_model = f
        # If distributions is not a list, it specifies the same distribution
        # for every dimension
        if not isinstance(distributions, list):
            distributions = [distributions for _ in range(dim)]

        # Setting the distribution to a string is a short form when
        # no parameters are given
        for d in range(dim):
            if isinstance(distributions[d], str):
                distributions[d] = (distributions[d],)

        self._prepare_distributions(distributions, a, b)
        self.f_evals = None
        self.gPCE = None
        self.pce_polys = None

    def set_grid(self, grid):
        self.grid = grid

    def set_reference_solution(self, reference_solution):
        self.reference_solution = reference_solution

    # From the user provided information about distributions, this function
    # creates the distributions list which contains Chaospy distributions
    def _prepare_distributions(self, distris, a: Sequence[float],
                               b: Sequence[float]):
        self.distributions = []
        self.distribution_infos = distris
        chaospy_distributions = []
        known_distributions = dict()
        for d in range(self.dim):
            distr_info = distris[d]
            distr_known = distr_info in known_distributions
            if distr_known:
                # Reuse the same distribution objects for multiple dimensions
                d_prev = known_distributions[distr_info]
                self.distributions.append(self.distributions[d_prev])
            else:
                known_distributions[distr_info] = d

            distr_type = distr_info[0]
            if distr_type == "Uniform":
                distr = cp.Uniform(a[d], b[d])
                chaospy_distributions.append(distr)
                if not distr_known:
                    self.distributions.append(UQDistribution.from_chaospy(distr))
            elif distr_type == "Triangle":
                midpoint = distr_info[1]
                assert isinstance(midpoint, float), "invalid midpoint"
                distr = cp.Triangle(a[d], midpoint, b[d])
                chaospy_distributions.append(distr)
                if not distr_known:
                    self.distributions.append(UQDistribution.from_chaospy(distr))
            elif distr_type == "Normal":
                mu = distr_info[1]
                sigma = distr_info[2]
                cp_distr = cp.Normal(mu=mu, sigma=sigma)
                chaospy_distributions.append(cp_distr)
                if not distr_known:
                    # The chaospy normal distribution does not work with big values
                    def pdf(x, _mu=mu, _sigma=sigma):
                        return sps.norm.pdf(x, loc=_mu, scale=_sigma)

                    def cdf(x, _mu=mu, _sigma=sigma):
                        return sps.norm.cdf(x, loc=_mu, scale=_sigma)

                    def ppf(x, _mu=mu, _sigma=sigma):
                        return sps.norm.ppf(x, loc=_mu, scale=_sigma)

                    self.distributions.append(UQDistribution(pdf, cdf, ppf))
            elif distr_type == "Laplace":
                mu = distr_info[1]
                scale = distr_info[2]
                cp_distr = cp.Laplace(mu=mu, scale=scale)
                chaospy_distributions.append(cp_distr)
                if not distr_known:
                    def pdf(x, _mu=mu, _scale=scale):
                        return sps.laplace.pdf(x, loc=_mu, scale=_scale)

                    def cdf(x, _mu=mu, _scale=scale):
                        return sps.laplace.cdf(x, loc=_mu, scale=_scale)

                    def ppf(x, _mu=mu, _scale=scale):
                        return sps.laplace.ppf(x, loc=_mu, scale=_scale)

                    self.distributions.append(UQDistribution(pdf, cdf, ppf))
            else:
                assert False, "Distribution not implemented: " + distr_type
        self.distributions_chaospy = chaospy_distributions
        self.distributions_joint = cp.J(*chaospy_distributions)
        self.all_uniform = all(k[0] == "Uniform" for k in known_distributions)
        self.a = a
        self.b = b

    def get_surplus_width(self, d: int, right_parent: float, left_parent: float) -> float:
        # Approximate the width with the probability
        cdf = self.distributions[d].cdf
        return cdf(right_parent) - cdf(left_parent)

    # This function exchanges the operation's function so that the adaptive
    # refinement can use a different function than the operation's function
    def set_function(self, f=None):
        if f is None:
            self.f = self.f_actual
            self.f_actual = None
        else:
            assert self.f_actual is None
            self.f_actual = self.f
            self.f = f

    def update_function(self, f):
        self.f = f

    def get_distributions(self):
        return self.distributions

    def get_distributions_chaospy(self):
        return self.distributions_chaospy

    # This function returns boundaries for distributions which have an infinite
    # domain, such as normal distribution
    def get_boundaries(self, tol: float) -> Tuple[Sequence[float], Sequence[float]]:
        assert 1.0 - tol < 1.0, "Tolerance is too small"
        a = []
        b = []
        for d in range(self.dim):
            dist = self.distributions[d]
            a.append(dist.ppf(tol))
            b.append(dist.ppf(1.0 - tol))
        return a, b

    def _set_pce_polys(self, polynomial_degrees):
        if self.pce_polys is not None and self.polynomial_degrees == polynomial_degrees:
            return
        self.polynomial_degrees = polynomial_degrees
        if not hasattr(polynomial_degrees, "__iter__"):
            self.pce_polys, self.pce_polys_norms = cp.orth_ttr(polynomial_degrees, self.distributions_joint, retall=True)
            return

        # Chaospy does not support different degrees for each dimension, so
        # the higher degree polynomials are removed afterwards
        polys, norms = cp.orth_ttr(max(polynomial_degrees), self.distributions_joint, retall=True)
        polys_filtered, norms_filtered = [], []
        for i, poly in enumerate(polys):
            max_exponents = [max(exps) for exps in poly.exponents.T]
            if any([max_exponents[d] > deg_max for d, deg_max in enumerate(polynomial_degrees)]):
                continue
            polys_filtered.append(poly)
            norms_filtered.append(norms[i])
        self.pce_polys = cp.Poly(polys_filtered)
        self.pce_polys_norms = norms_filtered

    def _scale_values(self, values):
        assert self.all_uniform, "Division by the domain volume should be used for uniform distributions only"
        div = 1.0 / np.prod([self.b[i] - v_a for i, v_a in enumerate(self.a)])
        return values * div

    def _set_nodes_weights_evals(self, combiinstance, scale_weights=False):
        self.nodes, self.weights = combiinstance.get_points_and_weights()
        assert len(self.nodes) == len(self.weights)
        if scale_weights:
            assert combiinstance.has_basis_grid(), "scale_weights should only be needed for basis grids"
            self.weights = self._scale_values(self.weights)
            # ~ self.f_evals = combiinstance.get_surplusses()
            # Surpluses are required here..
            self.f_evals = [self.f_model(coord) for coord in self.nodes]
        else:
            self.f_evals = [self.f_model(coord) for coord in self.nodes]

    def _get_combiintegral(self, combiinstance, scale_weights=False):
        integral = self.get_result()
        if scale_weights:
            assert combiinstance.has_basis_grid(), "scale_weights should only be needed for basis grids"
            return self._scale_values(integral)
        return integral

    def calculate_moment(self, combiinstance, k: int = None,
                         use_combiinstance_solution=True, scale_weights=False):
        if use_combiinstance_solution:
            mom = self._get_combiintegral(combiinstance, scale_weights=scale_weights)
            assert len(mom) == self.f_model.output_length()
            return mom
        self._set_nodes_weights_evals(combiinstance)
        vals = [self.f_evals[i] ** k * self.weights[i] for i in range(len(self.f_evals))]
        return sum(vals)

    def calculate_expectation(self, combiinstance, use_combiinstance_solution=True):
        return self.calculate_moment(combiinstance, k=1, use_combiinstance_solution=use_combiinstance_solution)

    @staticmethod
    def moments_to_expectation_variance(mom1: Sequence[float],
                                        mom2: Sequence[float]) -> Tuple[Sequence[float], Sequence[float]]:
        expectation = mom1
        variance = [mom2[i] - ex * ex for i, ex in enumerate(expectation)]
        for i, v in enumerate(variance):
            if v < 0.0:
                # When the variance is zero, it can be set to something negative
                # because of numerical errors
                variance[i] = -v
        return expectation, variance

    def calculate_expectation_and_variance(self, combiinstance, use_combiinstance_solution=True, scale_weights=False):
        if use_combiinstance_solution:
            integral = self._get_combiintegral(combiinstance, scale_weights=scale_weights)
            output_dim = len(integral) // 2
            expectation = integral[:output_dim]
            expectation_of_squared = integral[output_dim:]
        else:
            expectation = self.calculate_moment(combiinstance, k=1, use_combiinstance_solution=False)
            expectation_of_squared = self.calculate_moment(combiinstance, k=2, use_combiinstance_solution=False)
        return self.moments_to_expectation_variance(expectation, expectation_of_squared)

    def calculate_PCE(self, polynomial_degrees, combiinstance, restrict_degrees=False, use_combiinstance_solution=True, scale_weights=False):
        if use_combiinstance_solution:
            assert self.pce_polys is not None
            assert not restrict_degrees
            integral = self._get_combiintegral(combiinstance, scale_weights=scale_weights)
            num_polys = len(self.pce_polys)
            output_dim = len(integral) // num_polys
            coefficients = integral.reshape((num_polys, output_dim))
            self.gPCE = cp.poly.transpose(cp.poly.sum(self.pce_polys * coefficients.T, -1))
            return

        self._set_nodes_weights_evals(combiinstance)

        if restrict_degrees:
            # Restrict the polynomial degrees if in some dimension not enough points
            # are available
            # For degree deg, deg+(deg-1)+1 points should be available
            num_points = combiinstance.get_num_points_each_dim()
            polynomial_degrees = [min(polynomial_degrees, num_points[d] // 2) for d in range(self.dim)]

        self._set_pce_polys(polynomial_degrees)
        self.gPCE = cp.fit_quadrature(self.pce_polys, list(zip(*self.nodes)),
                                      self.weights, np.asarray(self.f_evals), norms=self.pce_polys_norms)

    def get_gPCE(self):
        return self.gPCE

    def get_expectation_PCE(self):
        if self.gPCE is None:
            assert False, "calculatePCE must be invoked before this method"
        return cp.E(self.gPCE, self.distributions_joint)

    def get_variance_PCE(self):
        if self.gPCE is None:
            assert False, "calculatePCE must be invoked before this method"
        return cp.Var(self.gPCE, self.distributions_joint)

    def get_expectation_and_variance_PCE(self):
        return self.get_expectation_PCE(), self.get_variance_PCE()

    def get_Percentile_PCE(self, q: float, sample: int = 10000):
        if self.gPCE is None:
            assert False, "calculatePCE must be invoked before this method"
        return cp.Perc(self.gPCE, q, self.distributions_joint, sample)

    def get_first_order_sobol_indices(self):
        if self.gPCE is None:
            assert False, "calculatePCE must be invoked before this method"
        return cp.Sens_m(self.gPCE, self.distributions_joint)

    def get_total_order_sobol_indices(self):
        if self.gPCE is None:
            assert False, "calculatePCE must be invoked before this method"
        return cp.Sens_t(self.gPCE, self.distributions_joint)

    # Returns a Function which can be passed to performSpatiallyAdaptiv
    # so that adapting is optimized for the k-th moment
    def get_moment_Function(self, k: int):
        if k == 1:
            return self.f
        return FunctionPower(self.f, k)

    def set_moment_Function(self, k: int):
        self.update_function(self.get_moment_Function(k))

    # Optimizes adapting for multiple moments at once
    def get_moments_Function(self, ks: Sequence[int]):
        return FunctionConcatenate([self.get_moment_Function(k) for k in ks])

    def set_moments_Function(self, ks: Sequence[int]):
        self.update_function(self.get_moments_Function(ks))

    def get_expectation_variance_Function(self):
        return self.get_moments_Function([1, 2])

    def set_expectation_variance_Function(self):
        self.update_function(self.get_expectation_variance_Function())

    # Returns a Function which can be passed to performSpatiallyAdaptiv
    # so that adapting is optimized for the PCE
    def get_PCE_Function(self, polynomial_degrees):
        self._set_pce_polys(polynomial_degrees)
        # self.f can change, so putting it to a local variable is important
        # ~ f = self.f
        # ~ polys = self.pce_polys
        # ~ funcs = [(lambda coords: f(coords) * polys[i](coords)) for i in range(len(polys))]
        # ~ return FunctionCustom(funcs)
        return FunctionPolysPCE(self.f, self.pce_polys, self.pce_polys_norms)

    def set_PCE_Function(self, polynomial_degrees):
        self.update_function(self.get_PCE_Function(polynomial_degrees))

    def get_pdf_Function(self):
        pdf = self.distributions_joint.pdf
        return FunctionCustom(lambda coords: float(pdf(coords)))

    def set_pdf_Function(self):
        self.update_function(self.get_pdf_Function())

    # Returns a Function which applies the PPF functions before evaluating
    # the problem function; it can be integrated without weighting
    def get_inverse_transform_Function(self, func=None):
        return FunctionInverseTransform(func or self.f, self.distributions)

    def set_inverse_transform_Function(self, func=None):
        self.update_function(self.get_inverse_transform_Function(func or self.f, self.distributions))


# UncertaintyQuantification extended for testing purposes
class UncertaintyQuantificationTesting(UncertaintyQuantification):
    # This function uses the quadrature provided by Chaospy.
    def calculate_PCE_chaospy(self, polynomial_degrees, num_quad_points):
        self._set_pce_polys(polynomial_degrees)
        nodes, weights = cp.generate_quadrature(num_quad_points,
                                                self.distributions_joint, rule="G")
        f_evals = [self.f(c) for c in zip(*nodes)]
        self.gPCE = cp.fit_quadrature(self.pce_polys, nodes, weights, np.asarray(f_evals), norms=self.pce_polys_norms)

    def calculate_expectation_and_variance_for_weights(self, nodes, weights):
        f_evals = np.array([self.f(c) for c in zip(*nodes)])
        f_evals_squared = np.array([v ** 2 for v in f_evals])
        expectation = np.inner(f_evals.T, weights)
        expectation_of_squared = np.inner(f_evals_squared.T, weights)
        return self.moments_to_expectation_variance(expectation, expectation_of_squared)

    def calculate_expectation_and_variance_reference(self, mode="ChaospyHalton", modeparams=None):
        if mode == "ChaospyHalton":
            num_points = modeparams or 2 ** 14
            nodes = self.distributions_joint.sample(num_points, rule="H")
            num_samples = len(nodes[0])
            assert num_points == num_samples
            w = 1.0 / num_samples
            weights = np.array([w for _ in range(num_samples)])
        elif mode == "ChaospyGauss":
            nodes, weights = cp.generate_quadrature(29,
                                                    self.distributions_joint, rule="G")
        elif mode == "StandardcombiGauss":
            if all([distr[0] == "Normal" for distr in self.distribution_infos]):
                expectations = [distr[1] for distr in self.distribution_infos]
                standard_deviations = [distr[2] for distr in self.distribution_infos]
                grid = GaussHermiteGrid(expectations, standard_deviations)
                # ~ combiinstance = StandardCombi(self.a, self.b, grid=grid, operation=self)
                combiinstance = StandardCombi(self.a, self.b, grid=grid)
                combiinstance.perform_combi(1, 4, self.get_expectation_variance_Function())
                combiinstance.print_resulting_combi_scheme(markersize=5)
                combiinstance.print_resulting_sparsegrid(markersize=10)
            elif self.all_uniform:

                grid = GaussLegendreGrid(self.a, self.b, self.dim)
                # ~ combiinstance = StandardCombi(self.a, self.b, grid=grid, operation=self)
                combiinstance = StandardCombi(self.a, self.b, grid=grid)
                combiinstance.perform_combi(1, 4, self.get_expectation_variance_Function())
                combiinstance.print_resulting_combi_scheme(markersize=5)
                combiinstance.print_resulting_sparsegrid(markersize=10)
            else:
                assert False, "Not implemented"
        else:
            assert False, mode
        return self.calculate_expectation_and_variance_for_weights(nodes, weights)

    def calculate_multiple_expectation_and_variance(self, solutions):
        evals = sorted(solutions)
        expectation_variances = []
        for k in evals:
            integral = solutions[k]
            output_dim = len(integral) // 2
            mom1 = integral[:output_dim]
            mom2 = integral[output_dim:]
            expectation, variance = self.moments_to_expectation_variance(mom1, mom2)
            expectation_variances.append((k, expectation, variance))
        return expectation_variances

    @staticmethod
    def sort_multiple_solutions(solutions):
        return [(num_evals, solutions[num_evals]) for num_evals in sorted(solutions)]

    def calculate_multiple_expectation_and_variance(self, solutions):
        expectation_variances = []
        for num_evals, integral in self.sort_multiple_solutions(solutions):
            output_dim = len(integral) // 2
            mom1 = integral[:output_dim]
            mom2 = integral[output_dim:]
            expectation, variance = self.moments_to_expectation_variance(mom1, mom2)
            expectation_variances.append((num_evals, expectation, variance))
        return expectation_variances

    def calculate_PCE_from_multiple(self, combiinstance, integrals):
        combiinstance.calculated_solution = integrals
        return self.calculate_PCE(None, combiinstance)


from scipy import integrate


class UQDistribution:
    def __init__(self, pdf, cdf, ppf):
        self.pdf = pdf
        self.cdf = cdf
        self.ppf = ppf
        self.cached_moments = [dict() for _ in range(2)]
        self.cache_gauss_quad = dict()
        # ~ self.cache_integrals = dict()

    @staticmethod
    def from_chaospy(cp_distr):
        # The inverse Rosenblatt transformation is the inverse cdf here
        return UQDistribution(cp_distr.pdf, cp_distr.cdf,
                              lambda x: float(cp_distr.inv(x)))

    def get_zeroth_moment(self, x1: float, x2: float):
        cache = self.cached_moments[0]
        if (x1, x2) in cache:
            return cache[(x1, x2)]
        moment_0 = self.cdf(x2) - self.cdf(x1)
        cache[(x1, x2)] = moment_0
        return moment_0

    def get_first_moment(self, x1: float, x2: float):
        cache = self.cached_moments[1]
        if (x1, x2) in cache:
            return cache[(x1, x2)]
        moment_1 = integrate.quad(lambda x: x * self.pdf(x), x1, x2,
                                  epsrel=10 ** -2, epsabs=np.inf)[0]
        cache[(x1, x2)] = moment_1
        return moment_1

    # Returns single-dimensional quadrature points and weights
    # for the high order grid
    def get_quad_points_weights(self, num_quad_points: int, cp_distribution) -> Tuple[Sequence[float], Sequence[float]]:
        cache = self.cache_gauss_quad
        if num_quad_points in cache:
            return cache[num_quad_points]
        (coords,), weights = cp.generate_quadrature(num_quad_points, cp_distribution, rule="G")
        cache[num_quad_points] = (coords, weights)
        return coords, weights

    # Calculates the weighted integral of an arbitrary function
    # between x1 and x2
    def calculate_integral(self, func, x1: float, x2: float):
        # ~ k = (func, x1, x2)
        # ~ cache = self.cache_integrals
        # ~ if k in cache:
        # ~ print("Cache match")
        # ~ return cache[k]
        integral = integrate.quad(lambda x: func(x) * self.pdf(x), x1, x2)[0]
        # ~ cache[k] = integral
        return integral
