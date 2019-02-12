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


class Integration(AreaOperation):
    def __init__(self, f, grid, reference_solution=None):
        self.f = f
        self.grid = grid
        self.reference_solution = reference_solution

    def evaluate_area(self, area, levelvector, componentgrid_info, refinement_container):
        partial_integral = componentgrid_info.coefficient * self.grid.integrate(self.f, levelvector, area.start, area.end)
        area.integral += partial_integral
        refinement_container.integral += partial_integral
        evaluations = np.prod(self.grid.levelToNumPoints(levelvector))
        return evaluations

    def count_unique_points(self):
        return True

    def area_preprocessing(self, area):
        area.set_integral(0.0)

    def get_global_error_estimate(self, refinement_container):
        if self.reference_solution is None:
            return None
        else:
            return abs((self.reference_solution - refinement_container.integral)/self.reference_solution)