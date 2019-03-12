import abc
import numpy as np

class GridOperation(object):
    def is_area_operation(self):
        return False
    @abc.abstractmethod
    def perform_operation(self, levelvector, refinementObjects=None):
        pass

    @abc.abstractmethod
    def set_errors(self, refinementObjects=None):
        pass

    # This method indicates whether we should only count unique points for the cost estimate (return True) or
    # if we should count points multiple times if they are contained in different component grids
    @abc.abstractmethod
    def count_unique_points(self):
        return False

    def area_preprocessing(self, area):
        pass

    def area_postprocessing(self, area):
        pass

    def post_processing(self):
        pass

    # This method can be used if there is a better global error estimate than the summation of the local surplusses
    def get_global_error_estimate(self, refinement):
        return None

class AreaOperation(GridOperation):
    def is_area_operation(self):
        return True

    def perform_operation(self, levelvector, refinementObjects=None):
        for area in enumerate(refinementObjects):
            self.evaluate_area(area, levelvector)

    @abc.abstractmethod
    def evaluate_area(self, area, levelvector, componentgrid_info, refinement_container):
        pass

from scipy.interpolate import interpn

class Integration(AreaOperation):
    def __init__(self, f, grid, dim,  reference_solution=None):
        self.f = f
        self.grid = grid
        self.reference_solution = reference_solution
        self.dim = dim

    def evaluate_area(self, area, levelvector, componentgrid_info, refinement_container, additional_info):
        partial_integral = componentgrid_info.coefficient * self.grid.integrate(self.f, levelvector, area.start, area.end)
        area.integral += partial_integral
        if refinement_container is not None:
            refinement_container.integral += partial_integral
        evaluations = np.prod(self.grid.levelToNumPoints(levelvector))
        return evaluations

    def evaluate_area_for_error_estimates(self, area, levelvector, componentgrid_info, refinement_container, additional_info):
        if additional_info.error_name == "extend_parent":
            assert additional_info.filter_area is None
            extend_parent_new = self.grid.integrate(self.f, levelvector, area.start, area.end)
            if area.parent_info.extend_parent_integral is None:
                area.parent_info.extend_parent_integral = 0.0
            area.parent_info.extend_parent_integral += extend_parent_new * componentgrid_info.coefficient
            return np.prod(self.grid.levelToNumPoints(levelvector))
        elif additional_info.error_name == "extend_error_correction":
            assert additional_info.filter_area is None
            if area.switch_to_parent_estimation:
                extend_parent_new = self.grid.integrate(self.f, levelvector, area.start, area.end)
                if area.parent_info.extend_error_correction is None:
                    area.parent_info.extend_error_correction = area.integral
                area.parent_info.extend_error_correction -= extend_parent_new * componentgrid_info.coefficient
                return np.prod(self.grid.levelToNumPoints(levelvector))
            else:
                return 0
        elif additional_info.error_name == "split_no_filter":
                assert additional_info.filter_area is None
                split_parent_new = self.grid.integrate(self.f, levelvector, area.start, area.end)
                if additional_info.target_area.parent_info.split_parent_integral is None:
                    additional_info.target_area.parent_info.split_parent_integral = 0.0
                additional_info.target_area.parent_info.split_parent_integral += split_parent_new * componentgrid_info.coefficient
                return np.prod( self.grid.levelToNumPoints(levelvector))
        else:
            assert additional_info.filter_area is not None
            if not additional_info.interpolate:  # use filtering approach
                self.grid.setCurrentArea(area.start, area.end, levelvector)
                points, weights = self.grid.get_points_and_weights()
                integral = 0.0
                num_points = 0
                for i, p in enumerate(points):
                    if self.point_in_area(p, additional_info.filter_area):
                        integral += self.f(p) * weights[i] * self.get_point_factor(p, additional_info.filter_area, area)
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
                interpolated_values = self.interpolate_points(mesh_points_grid, points)

                integral += sum(
                    [interpolated_values[i] * weights[i] for i in range(len(interpolated_values))])

                # calculate all mesh points
                mesh_points = list(
                    zip(*[g.ravel() for g in np.meshgrid(*[mesh_points_grid[d] for d in range(self.dim)])]))

                # count the number of mesh points that fall into the filter area
                for p in mesh_points:
                    if self.point_in_area(p, additional_info.filter_area) and self.grid.point_not_zero(p):
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


    def get_best_fit(self, area):
        old_value = area.parent_info.split_parent_integral
        new_value = area.parent_info.split_parent_integral2
        if old_value is None:
            area.parent_info.split_parent_integral = new_value
        else:
            if new_value is None or abs(area.integral - old_value) < abs(area.integral - new_value):
                pass
            else:
                area.parent_info.split_parent_integral = area.parent_info.split_parent_integral2

    def initialize_error_estimates(self,area):
        area.parent_info.split_parent_integral = None
        area.parent_info.split_parent_integral2 = None
        area.parent_info.extend_parent_integral = None
        if area.parent_info.benefit_split is None:
            area.parent_info.num_points_extend_parent = None
        if area.parent_info.benefit_extend is None:
            area.parent_info.num_points_split_parent = None
        area.parent_info.num_points_reference = None
        area.parent_info.extend_error_correction = None

    def initialize_split_error(self, area):
        area.parent_info.split_parent_integral = None
        area.parent_info.split_parent_integral2 = None
        if area.parent_info.benefit_extend is None:
            area.parent_info.num_points_split_parent = None

    def get_previous_value_from_split_parent(self, area):
        self.get_best_fit(area)
        area.parent_info.previous_value = area.parent_info.split_parent_integral

    def count_unique_points(self):
        return True

    def area_preprocessing(self, area):
        area.set_integral(0.0)

    def get_global_error_estimate(self, refinement_container):
        if self.reference_solution is None:
            return None
        else:
            return abs((self.reference_solution - refinement_container.integral)/self.reference_solution)

    def area_postprocessing(self, area):
        area.value = area.integral

    def get_sum_sibling_value(self, area):
        area.sum_siblings = 0.0
        i = 0
        for child in area.parent_info.parent.children:
            if child.integral is not None:
                area.sum_siblings += child.integral
                i += 1
        assert i == 2 ** self.dim  # we always have 2**dim children

    def set_extend_benefit(self, area):
        if area.parent_info.benefit_extend is not None:
            return
        if area.switch_to_parent_estimation:
            comparison = area.sum_siblings
            num_comparison = area.evaluations * 2 ** self.dim
        else:
            comparison = area.integral
            num_comparison = area.evaluations
        assert num_comparison > area.parent_info.num_points_split_parent or area.switch_to_parent_estimation

        error_extend = abs((area.parent_info.split_parent_integral - comparison) / (abs(comparison) + 10 ** -100))
        if not self.grid.is_high_order_grid():
            area.parent_info.benefit_extend = error_extend * (area.parent_info.num_points_split_parent - area.parent_info.num_points_reference)
        else:
            area.parent_info.benefit_extend = error_extend * area.parent_info.num_points_split_parent

    def set_split_benefit(self, area):
        if area.parent_info.benefit_split is not None:
            return
        if area.switch_to_parent_estimation:
            num_comparison = area.evaluations * 2 ** self.dim
        else:
            num_comparison = area.evaluations
        assert num_comparison > area.parent_info.num_points_extend_parent

        if self.grid.boundary:
            assert area.parent_info.num_points_split_parent > 0
        error_split = abs((area.parent_info.extend_parent_integral - area.integral) / (abs(area.integral) + 10 ** -100))
        if not self.grid.is_high_order_grid():
            area.parent_info.benefit_split = error_split * (area.parent_info.num_points_extend_parent - area.parent_info.num_points_reference)
        else:
            area.parent_info.benefit_split = error_split * area.parent_info.num_points_extend_parent

    def set_extend_error_correction(self, area):
        if area.switch_to_parent_estimation:
            area.parent_info.extend_error_correction *= area.parent_info.num_points_split_parent

    def point_in_area(self, point, area):
        for d in range(self.dim):
            if point[d] < area.start[d] or point[d] > area.end[d]:
                return False
        return True

    def get_point_factor(self, point, area, area_parent):
        factor = 1.0
        for d in range(self.dim):
            if (point[d] == area.start[d] or point[d] == area.end[d]) and not (
                    point[d] == area_parent.start[d] or point[d] == area_parent.end[d]):
                factor /= 2.0
        return factor

    def initialize_global_value(self, refinement):
        refinement.integral = 0.0

    # interpolates the cell at the subcell edge points and evaluates the integral based on the trapezoidal rule
    def compute_subcell_with_interpolation(self, cell, subcell, coefficient, refinement_container):
        #print("Cell and subcell", cell, subcell)
        start_subcell = subcell.start
        end_subcell = subcell.end
        start_cell = cell.start
        end_cell = cell.end
        subcell_points = list(zip(*[g.ravel() for g in np.meshgrid(*[[start_subcell[d], end_subcell[d]] for d in range(self.dim)])]))
        corner_points_grid = [[start_cell[d], end_cell[d]] for d in range(self.dim)]
        interpolated_values = self.interpolate_points(corner_points_grid, subcell_points)
        width = np.prod(np.array(end_subcell) - np.array(start_subcell))
        factor = 0.5**self.dim * width
        integral = 0.0
        for p in interpolated_values:
            integral += p * factor
        subcell.cell_dict[subcell.get_key()].sub_integrals.append((integral, coefficient))
        subcell.integral += integral * coefficient
        if refinement_container is not None:
            refinement_container.integral += integral * coefficient

    # interpolates mesh_points_grid at the given  evaluation_points using bilinear interpolation
    def interpolate_points(self, mesh_points_grid, evaluation_points):
        # constructing all points from mesh definition
        mesh_points = list(zip(*[g.ravel() for g in np.meshgrid(*[mesh_points_grid[d] for d in range(self.dim)])]))

        # calculate function values at mesh points and transform  correct data structure for scipy
        values = np.array([self.f(p) if self.grid.point_not_zero(p) else 0.0 for p in mesh_points])
        values = values.reshape(*[len(mesh_points_grid[d]) for d in reversed(range(self.dim))])
        values = np.transpose(values)

        # interpolate evaluation points from mesh points with bilinear interpolation
        interpolated_values = interpn(mesh_points_grid, values, evaluation_points, method='linear')
        return interpolated_values

    # returns the absolute error in the integrals of the given area object and its twin in the d'th dimension
    def get_twin_error(self, d, area):
        return abs(area.integral - area.twins[d].integral)