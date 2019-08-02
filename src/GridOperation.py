from numpy import linalg as LA
from Grid import *
from BasisFunctions import *

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
        self.dict_integral = {}
        self.dict_points = {}

    def evaluate_area(self, area, levelvector, componentgrid_info, refinement_container, additional_info):
        partial_integral = componentgrid_info.coefficient * self.grid.integrate(self.f, levelvector, area.start, area.end)
        area.integral += partial_integral
        evaluations = np.prod(self.grid.levelToNumPoints(levelvector))
        if refinement_container is not None:
            refinement_container.integral += partial_integral
        return evaluations

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
                    area.parent_info.extend_error_correction = np.array(area.integral)
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


    def get_best_fit(self, area, norm):
        old_value = area.parent_info.split_parent_integral
        new_value = area.parent_info.split_parent_integral2
        if old_value is None:
            area.parent_info.split_parent_integral = np.array(new_value)
        else:
            if new_value is None or LA.norm(abs(area.integral - old_value), norm) < LA.norm(abs(area.integral - new_value), norm):
                pass
            else:
                area.parent_info.split_parent_integral = np.array(area.parent_info.split_parent_integral2)

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

    def initialize_error(self, area, error_name):
        if error_name == "split_parent":
            area.parent_info.split_parent_integral = None
        if error_name == "split_parent2":
            area.parent_info.split_parent_integral2 = None
        if error_name == "extend_parent":
            area.parent_info.extend_parent_integral = None

    def initialize_point_numbers(self, area, error_name):
        if error_name == "split_parent":
            area.parent_info.num_points_split_parent = None
        if error_name == "split_parent2":
            area.parent_info.num_points_split_parent = None
        if error_name == "extend_parent":
            area.parent_info.num_points_extend_parent = None

    def get_previous_value_from_split_parent(self, area):
        area.parent_info.previous_value = area.parent_info.split_parent_integral

    def count_unique_points(self):
        return True

    def area_preprocessing(self, area):
        area.set_integral(0.0)

    def get_global_error_estimate(self, refinement_container, norm):
        if self.reference_solution is None:
            return None
        else:
            return LA.norm(abs((self.reference_solution - refinement_container.integral)/self.reference_solution), norm)

    def area_postprocessing(self, area):
        area.value = np.array(area.integral)

    def get_sum_sibling_value(self, area):
        area.sum_siblings = 0.0
        i = 0
        for child in area.parent_info.parent.children:
            if child.integral is not None:
                area.sum_siblings += child.integral
                i += 1
        assert i == 2 ** self.dim  # we always have 2**dim children

    def set_extend_benefit(self, area, norm):
        if area.parent_info.benefit_extend is not None:
            return
        if area.switch_to_parent_estimation:
            comparison = area.sum_siblings
            num_comparison = area.evaluations * 2 ** self.dim
        else:
            comparison = area.integral
            num_comparison = area.evaluations
        assert num_comparison > area.parent_info.num_points_split_parent or area.switch_to_parent_estimation

        error_extend = LA.norm(abs((area.parent_info.split_parent_integral - comparison) / (abs(comparison) + 10 ** -100)), norm)
        if not self.grid.is_high_order_grid():
            area.parent_info.benefit_extend = error_extend * (area.parent_info.num_points_split_parent - area.parent_info.num_points_reference)
        else:
            area.parent_info.benefit_extend = error_extend * area.parent_info.num_points_split_parent

    def set_split_benefit(self, area, norm):
        if area.parent_info.benefit_split is not None:
            return
        if area.switch_to_parent_estimation:
            num_comparison = area.evaluations * 2 ** self.dim
        else:
            num_comparison = area.evaluations
        assert num_comparison > area.parent_info.num_points_extend_parent

        if self.grid.boundary:
            assert area.parent_info.num_points_split_parent > 0
        error_split = LA.norm(abs((area.parent_info.extend_parent_integral - area.integral) / (abs(area.integral) + 10 ** -100)), norm)
        if not self.grid.is_high_order_grid():
            area.parent_info.benefit_split = error_split * (area.parent_info.num_points_extend_parent - area.parent_info.num_points_reference)
        else:
            area.parent_info.benefit_split = error_split * area.parent_info.num_points_extend_parent

    def set_extend_error_correction(self, area, norm):
        if area.switch_to_parent_estimation:
            area.parent_info.extend_error_correction = LA.norm(area.parent_info.extend_error_correction, norm) * area.parent_info.num_points_split_parent

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
        return Interpolation.interpolate_points(self.f, self.dim, self.grid, mesh_points_grid, evaluation_points)

    def print_evaluation_output(self, refinement):
        combi_integral = refinement.integral
        if len(combi_integral) == 1:
            combi_integral = combi_integral[0]
        print("combiintegral:", combi_integral)

    def calculate_operation_dimension_wise(self, gridPointCoordsAsStripes, grid_point_levels, component_grid, start, end, reuse_old_values):
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
                integral -= self.subtract_contributions(modification_points, previous_points_coarsened,
                                                        gridPointCoordsAsStripes)
                integral += self.get_new_contributions(modification_points, gridPointCoordsAsStripes)
        else:
            self.grid_surplusses.set_grid(gridPointCoordsAsStripes, grid_point_levels)
            self.grid.set_grid(gridPointCoordsAsStripes, grid_point_levels)
            integral = self.grid.integrate(self.f, component_grid.levelvector, start, end)
        self.refinement_container.integral += integral * component_grid.coefficient
        self.dict_integral[tuple(component_grid.levelvector)] = np.array(integral)
        self.dict_points[tuple(component_grid.levelvector)] = np.array(gridPointCoordsAsStripes)
        return integral

    def compute_error_estimates_dimension_wise(self, gridPointCoordsAsStripes, grid_point_levels, children_indices, component_grid):
        self.grid_surplusses.set_grid(gridPointCoordsAsStripes, grid_point_levels)
        self.grid.set_grid(gridPointCoordsAsStripes, grid_point_levels)
        self.calculate_surplusses(gridPointCoordsAsStripes, children_indices, component_grid)

    def init_dimension_wise(self, grid, grid_surplusses, f, refinement_container, lmin, lmax, a, b, version = 2):
        self.grid = grid
        self.grid_surplusses = grid_surplusses
        self.f = f
        self.refinement_container = refinement_container
        self.version = version
        self.lmin = lmin
        self.lmax = lmax
        self.a = a
        self.b = b

    def initialize_refinement_container_dimension_wise(self, refinement_container):
        refinement_container.integral = 0.0
    # This method calculates the surplus error estimates for a point by calculating dim-1 dimensional slices
    # through the domain along the child coordinates. We always calculate the 1-dimensional surplus for every point
    # on this slice.
    def calculate_surplusses(self, grid_points, children_indices, component_grid):
        tol = 10**-14
        for d in range(0, self.dim):
            k=0
            refinement_dim = self.refinement_container.get_refinement_container_for_dim(d)
            if isinstance(self.grid, GlobalBSplineGrid) or isinstance(self.grid, GlobalLagrangeGrid):
                grid_values = np.empty((self.f.output_length(), np.prod(self.grid.numPoints)))
                hierarchization_operator = HierarchizationLSG(self.grid)
                points, weights = self.grid.get_points_and_weights()
                surplusses_1d = hierarchization_operator.hierarchize_poles_for_dim(grid_values, self.grid.numPoints, self.f, d, True)
                surplus_pole = np.zeros((self.f.output_length(), self.grid.numPoints[d]))
                stride = int(np.prod(self.grid.numPoints[d+1:]))
                for j in range(self.grid.numPoints[d]):
                    i = j * stride
                    while i < np.prod(self.grid.numPoints):
                        surplus_pole[:,j] += np.sum(abs(surplusses_1d[:,i:i+stride])) #* weights[i:i+stride]))
                        i += stride * self.grid.numPoints[d]
                #toDo sum up pole surplusses and use below
                #print("surplus pole", surplus_pole, surplusses_1d, stride)
            for child_info in children_indices[d]:
                left_parent = child_info.left_parent
                right_parent = child_info.right_parent
                child = child_info.child
                if isinstance(self.grid, GlobalBSplineGrid) or isinstance(self.grid, GlobalLagrangeGrid):
                    index_child = grid_points[d].index(child) - int(not(self.grid.boundary))
                    volume = surplus_pole[:, index_child] / np.prod(self.grid.numPoints) * self.grid.numPoints[d] * self.grid.weights[d][index_child]
                    evaluations = np.prod(self.grid.numPoints) / self.grid.numPoints[d]
                else:
                    volume, evaluations = self.sum_up_volumes_for_point(child_info=child_info, grid_points=grid_points, d=d)
                #print("surplus pole", surplus_pole[1], grid_values[d], child)
                #print(surplus_pole[grid_points[d].index(child)])


                k_old = 0
                for i in range(refinement_dim.size() ):
                    if refinement_dim.get_object(i).start >= left_parent * (1 - tol):
                        k_old = i
                        break
                k = k_old
                refine_obj = refinement_dim.get_object(k)
                if not (refine_obj.start >= left_parent * (1 - tol) and refine_obj.end <= right_parent * (1 + tol)):
                    for child_info in children_indices[d]:
                        print(child_info.left_parent, child_info.child, child_info.right_parent)
                #print(refine_obj.start, refine_obj.end, left_parent, right_parent)
                assert refine_obj.start >= left_parent * (1 - tol) and refine_obj.end <= right_parent * (1 + tol)
                max_level = 1
                while k < refinement_dim.size():
                    refine_obj = refinement_dim.get_object(k)
                    if refine_obj.start >= right_parent * (1 - tol):
                        break
                    #refine_obj.print()
                    #print("Right parent", right_parent)
                    assert refine_obj.end <= right_parent * (1 + tol)
                    k += 1
                    max_level = max(max_level, max(refine_obj.levels))
                for i in range(k_old, k):
                    refine_obj = refinement_dim.get_object(i)
                    num_area_in_support = (k-k_old)
                    fraction_of_support = (refine_obj.end - refine_obj.start)/(right_parent - left_parent)
                    modified_volume = volume/num_area_in_support ** 2 #/ 2**(max_level - log2((self.b[d] - self.a[d])/(right_parent - left_parent))) #/  (num_area_in_support)**2
                    assert fraction_of_support <= 1
                    #print(modified_volume, left_parent, child, right_parent, refine_obj.start, refine_obj.end, num_area_in_support, evaluations)
                    #if not self.combischeme.has_forward_neighbour(component_grid.levelvector):
                    #print(volume)
                    refine_obj.add_volume(modified_volume * component_grid.coefficient)
                    #refine_obj.add_evaluations(evaluations / num_area_in_support * component_grid.coefficient)
                    #assert component_grid.coefficient == 1
                     #* component_grid.coefficient)
                    #print("Dim:", d, refine_obj.start, refine_obj.end, refine_obj.volume, refine_obj.evaluations, child, left_parent, right_parent, volume, modified_volume)

                    '''
                    if refine_obj.start >= left_parent and refine_obj.end <= child: #and not child_info.has_left_child:
                        width_refinement = refine_obj.end - refine_obj.start
                        width_basis = right_parent - left_parent
                        refine_obj.add_volume(modified_volume)  # * width_refinement/ width_basis)

                    elif refine_obj.start >= child and refine_obj.end <= right_parent: #and not child_info.has_right_child:
                        width_refinement = refine_obj.end - refine_obj.start
                        width_basis = right_parent - left_parent
                        refine_obj.add_volume(modified_volume)  # * width_refinement/ width_basis)
                    else:
                        break
                    '''
                k = min(k, refinement_dim.size() - 1)

                '''
                if not child_info.has_right_child:
                    child_info.right_refinement_object.add_volume(volume / 2.0)
                    child_info.right_refinement_object.add_evaluations(evaluations / 2.0)
                if not child_info.has_left_child:
                    child_info.left_refinement_object.add_volume(volume/2.0)
                    child_info.left_refinement_object.add_evaluations(evaluations / 2.0)
                '''

    # Sum up the 1-d surplusses along the dim-1 dimensional slice through the point child in dimension d.
    #  The surplusses are calculated based on the left and right parents.
    def sum_up_volumes_for_point(self, child_info, grid_points, d):
        #print(grid_points)
        child = child_info.child
        left_parent = child_info.left_parent
        right_parent = child_info.right_parent
        left_parent_of_left_parent = child_info.left_parent_of_left_parent
        right_parent_of_right_parent = child_info.right_parent_of_right_parent
        volume = 0.0
        assert right_parent > child > left_parent

        #npt.assert_almost_equal(right_parent - child, child - left_parent, decimal=12)

        for p in grid_points[d]:
            if isclose(p, left_parent):
                left_parent = p
            if isclose(p, right_parent):
                right_parent = p
        index_right_parent = grid_points[d].index(right_parent) - 1 * int(not self.grid.boundary)
        index_left_parent = grid_points[d].index(left_parent) - 1 * int(not self.grid.boundary)

        left_parent_in_grid = self.grid_surplusses.boundary or not(isclose(left_parent, self.a[d]))
        right_parent_in_grid = self.grid_surplusses.boundary or not(isclose(right_parent, self.b[d]))
        # avoid evaluating on boundary points if grids has none
        if left_parent_in_grid:
            factor_left_parent = (right_parent - child)/(right_parent - left_parent)
            points_left_parent = list(zip(*[g.ravel() for g in np.meshgrid(*[self.grid_surplusses.coords[d2]if d != d2 else [left_parent] for d2 in range(self.dim)])]))
        if right_parent_in_grid:
            factor_right_parent = (child - left_parent)/(right_parent - left_parent)
            points_right_parent = list(zip(*[g.ravel() for g in np.meshgrid(*[self.grid_surplusses.coords[d2] if d != d2 else [right_parent] for d2 in range(self.dim)])]))
        points_children = list(zip(*[g.ravel() for g in np.meshgrid(*[self.grid_surplusses.coords[d2] if d != d2 else [child] for d2 in range(self.dim)])]))
        indices = list(zip(*[g.ravel() for g in np.meshgrid(*[range(len(self.grid_surplusses.coords[d2])) if d != d2 else None for d2 in range(self.dim)])]))
        for i in range(len(points_children)):
            index = indices[i]
            factor = np.prod([self.grid_surplusses.weights[d2][index[d2]] if d2 != d else 1 for d2 in range(self.dim)])
            #factor2 = np.prod([self.grid.weights[d2][index[d2]]  if d2 != d else self.grid.weights[d2][index_child] for d2 in range(self.dim)])
            if factor != 0:
                exponent = 1# if not self.do_high_order else 2
                #if factor2 != 0:
                value = self.f(points_children[i])
                #print(points_children[i], self.f.f_dict.keys())
                # avoid evaluating on boundary points if grids has none
                assert (tuple(points_children[i]) in self.f.f_dict)

                if left_parent_in_grid:
                    if self.grid_surplusses.modified_basis and not right_parent_in_grid:
                        assert points_left_parent[i] in self.f.f_dict or self.grid.weights[d][index_left_parent] == 0

                        left_of_left_parent = list(points_left_parent[i])
                        left_of_left_parent[d] = left_parent_of_left_parent
                        #print("Left of left:", left_of_left_parent, points_left_parent[i])
                        #value = (2 * self.f(points_children[i]) - self.f(points_left_parent[i]))/2
                        #assert (tuple(points_left_parent[i]) in self.f.f_dict)

                        if isclose(left_of_left_parent[d], self.a[d]):
                            value = self.f(points_children[i]) - self.f(points_left_parent[i])
                        else:
                            m = (self.f(tuple(left_of_left_parent)) - self.f(points_left_parent[i])) / (
                                        left_parent_of_left_parent - left_parent)
                            previous_value_at_child = m * (child - left_parent) + self.f(points_left_parent[i])
                            value = self.f(points_children[i]) - previous_value_at_child
                            #print("Hey", m, previous_value_at_child, value, (self.f(tuple(left_of_left_parent)) - self.f(points_left_parent[i])), (left_of_left_parent - left_parent))

                            assert(tuple(left_of_left_parent) in self.f.f_dict)

                    else:
                        assert points_left_parent[i] in self.f.f_dict or self.grid.weights[d][index_left_parent] == 0

                        value -= factor_left_parent * self.f(points_left_parent[i])
                if right_parent_in_grid:
                    if self.grid_surplusses.modified_basis and not left_parent_in_grid:
                        assert points_right_parent[i] in self.f.f_dict or self.grid.weights[d][index_right_parent] == 0

                        right_of_right_parent = list(points_right_parent[i])
                        right_of_right_parent[d] = right_parent_of_right_parent
                        #print("Right of right:", right_of_right_parent, points_right_parent[i])
                        #value = (2 * self.f(points_children[i]) - self.f(points_right_parent[i]))/2
                        #assert (tuple(points_right_parent[i]) in self.f.f_dict)
                        if isclose(right_of_right_parent[d], self.b[d]):
                            value = self.f(points_children[i]) - self.f(points_right_parent[i])
                        else:
                            m = (self.f(tuple(right_of_right_parent)) - self.f(points_right_parent[i]))  / (right_parent_of_right_parent - right_parent)
                            previous_value_at_child = m * (child - right_parent) + self.f(points_right_parent[i])
                            value = self.f(points_children[i]) - previous_value_at_child
                            #print("Hey", m, previous_value_at_child, value, (self.f(tuple(right_of_right_parent)) - self.f(points_right_parent[i])), (right_of_right_parent - right_parent))
                            assert(tuple(right_of_right_parent) in self.f.f_dict)
                    else:
                        #print(points_right_parent[i], self.f.f_dict.keys())
                        assert points_right_parent[i] in self.f.f_dict or self.grid.weights[d][index_right_parent] == 0
                        value -= factor_right_parent * self.f(points_right_parent[i])
                volume += factor * abs(value) * (right_parent - left_parent)**exponent
        if self.version == 0 or self.version == 2:
            evaluations = len(points_children) #* (1 + int(left_parent_in_grid) + int(right_parent_in_grid))
        else:
            evaluations = 0
        return abs(volume), evaluations

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
            #get removed points for dimension d
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
                points_for_slice = list([point[0],point[2]])
                # remove boundary points if contained if grid has no boundary points
                if not self.grid.boundary:
                    points_for_slice = [p for p in points_for_slice if not(isclose(p, self.a[d]) or isclose(p, self.b[d]))]
                integral += self.calc_slice_through_points(points_for_slice, old_points, d, modification_points, subtract_contribution=True, dict=dict_weights_fine)
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
        points = list(zip(*[g.ravel() for g in np.meshgrid(*[self.grid_surplusses.coords[d2] if d != d2 else points_for_slice for d2 in range(self.dim)])]))
        indices = list(zip(*[g.ravel() for g in np.meshgrid(*[range(len(self.grid_surplusses.coords[d2])) if d != d2 else positions for d2 in range(self.dim)])]))
        for i in range(len(points)):
            # index of current point in grid_points grid
            index = indices[i]
            #point coordinates of current point
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
                factor = (weight - weight_fine)/number_of_dimensions_that_intersect
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
            assert(factor >= 0)
            integral += self.f(current_point) * factor
        return integral

class Interpolation(Integration):
    # interpolates mesh_points_grid at the given  evaluation_points using bilinear interpolation
    @staticmethod
    def interpolate_points(f, dim, grid, mesh_points_grid, evaluation_points):
        # constructing all points from mesh definition
        mesh_points = get_cross_product(mesh_points_grid)

        function_value_dim = len(f(np.ones(dim)*0.5))

        # calculate function values at mesh points and transform  correct data structure for scipy
        values = np.array([f(p) if grid.point_not_zero(p) else np.zeros(function_value_dim) for p in mesh_points])
        interpolated_values_array = []
        for d in range(function_value_dim):
            values_1D = np.asarray([value[d] for value in values])

            values_1D = values_1D.reshape(*[len(mesh_points_grid[d]) for d in (range(dim))])

            # interpolate evaluation points from mesh points with bilinear interpolation
            interpolated_values = interpn(mesh_points_grid, values_1D, evaluation_points, method='linear')

            interpolated_values = np.asarray([[value] for value in interpolated_values])
            interpolated_values_array.append(interpolated_values)
        return np.hstack(interpolated_values_array)
