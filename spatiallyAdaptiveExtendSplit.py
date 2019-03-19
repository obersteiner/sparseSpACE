from spatiallyAdaptiveBase import *
from GridOperation import *

class SpatiallyAdaptiveExtendScheme(SpatiallyAdaptivBase):
    def __init__(self, a, b, number_of_refinements_before_extend=1, grid=None, no_initial_splitting=False,
                 version=0, dim_adaptive=False, automatic_extend_split=False, operation=None, norm=np.inf):
        # there are three different version that coarsen grids slightly different
        # version 0 coarsen as much as possible while extending and adding only new points in regions where it is supposed to
        # version 1 coarsens less and also adds moderately many points in non refined regions which might result in a more balanced configuration
        # version 2 coarsen fewest and adds a bit more points in non refinded regions but very similar to version 1
        assert 2 >= version >= 0
        self.version = version
        SpatiallyAdaptivBase.__init__(self, a=a, b=b, grid=grid, operation=operation, norm=norm)
        self.noInitialSplitting = no_initial_splitting
        self.numberOfRefinementsBeforeExtend = number_of_refinements_before_extend
        self.refinements_for_recalculate = 100
        self.dim_adaptive = dim_adaptive
        self.automatic_extend_split = automatic_extend_split

    def interpolate_points(self, interpolation_points, component_grid):
        point_assignements = self.get_points_assignement_to_areas(interpolation_points)
        dict_point_interpolation_values = {}
        f_value_array_length = len(self.f([0.5]*self.dim))
        for area, contained_points in point_assignements:
            num_sub_diagonal = (self.lmax[0] + self.dim - 1) - np.sum(component_grid.levelvector)
            coarsened_levelvector, do_compute  = self.coarsen_grid(component_grid.levelvector, area, num_sub_diagonal)
            if do_compute:
                #print(coarsened_levelvector, contained_points, area.start, area.end)
                self.grid.setCurrentArea(start=area.start, end=area.end, levelvec=coarsened_levelvector)
                interpolated_values = Interpolation.interpolate_points(self.f, self.dim, self.grid, self.grid.coordinate_array, contained_points)
                for p, value in zip(contained_points, interpolated_values):
                    dict_point_interpolation_values[tuple(p)] = value
            else:
                for p in contained_points:
                    dict_point_interpolation_values[tuple(p)] = np.zeros(f_value_array_length)

        final_integrals = np.zeros((len(interpolation_points),f_value_array_length))
        for i, p in enumerate(interpolation_points):
            final_integrals[i] = dict_point_interpolation_values[tuple(p)]
        return final_integrals

    def get_points_assignement_to_areas(self, points):
        #print(points)
        return self.get_points_in_areas_recursive(self.root_cell, points)

    def get_points_in_areas_recursive(self, area, points):
        if area.children != []:
            point_assignements = []
            for sub_area in area.children:
                contained_points = sub_area.subset_of_contained_points(points)
                point_assignements.extend(self.get_points_in_areas_recursive(sub_area, contained_points))
                points = set(points) - set(contained_points)
                if len(points) == 0:
                    break
            return point_assignements
        else:
            return [(area, points)]



    # draw a visual representation of refinement tree
    def draw_refinement(self, filename=None):
        plt.rcParams.update({'font.size': 32})
        dim = self.dim
        if dim > 2:
            print("Refinement can only be printed in 2D")
            return
        fig, ax = plt.subplots(figsize=(20, 20))

        self.add_refinment_to_figure_axe(ax)

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
        plt.show()
        return fig

    def add_refinment_to_figure_axe(self, ax, linewidth=1):
        for i in self.refinement.get_objects():
            startx = i.start[0]
            starty = i.start[1]
            endx = i.end[0]
            endy = i.end[1]
            ax.add_patch(
                patches.Rectangle(
                    (startx, starty),
                    endx - startx,
                    endy - starty,
                    fill=False,  # remove background,
                    alpha=1,
                    linewidth=linewidth, visible=True
                )
            )

    # returns the points of a single component grid with refinement
    def get_points_component_grid(self, levelvec, numSubDiagonal):
        assert (numSubDiagonal < self.dim)
        points_array = []
        for area in self.refinement.get_objects():
            start = area.start
            end = area.end
            level_interval, do_compute = self.coarsen_grid(levelvec, area, numSubDiagonal)
            self.grid.setCurrentArea(start, end, level_interval)
            points = self.grid.getPoints()
            points_array.extend(points)
        return points_array

    def get_points_and_weights_component_grid(self, levelvec, numSubDiagonal):
        assert (numSubDiagonal < self.dim)
        points_array = []
        weights_array = []
        for area in self.refinement.get_objects():
            start = area.start
            end = area.end
            level_interval, do_compute = self.coarsen_grid(levelvec, area, numSubDiagonal)
            self.grid.setCurrentArea(start, end, level_interval)
            points, weights = self.grid.get_points_and_weights()
            points_array.extend(points)
            weights_array.extend(weights)
        return points_array, weights_array

    # returns the points of a single component grid with refinement
    def get_points_component_grid_not_null(self, levelvec, numSubDiagonal):
        assert (numSubDiagonal < self.dim)
        array2 = []
        for area in self.refinement.get_objects():
            start = area.start
            end = area.end
            level_interval, do_compute = self.coarsen_grid(levelvec, area, numSubDiagonal)
            if do_compute:
                self.grid.setCurrentArea(start, end, level_interval)
                points = self.grid.getPoints()
                array2.extend(points)
                # print("considered", levelvec, level_interval, area.start, area.end, area.coarseningValue)
            # else:
            # print("not considered", levelvec, level_interval, area.start, area.end, area.coarseningValue)
        return array2

    # optimized adaptive refinement refine multiple cells in close range around max variance (here set to 10%)
    def coarsen_grid(self, levelvector, area, num_sub_diagonal):
        start = area.start
        end = area.end
        coarsening = area.coarseningValue
        temp = list(levelvector)
        coarsening_save = coarsening
        area_is_null = False
        if self.version == 0:

            maxLevel = max(temp)
            temp2 = list(reversed(sorted(list(temp))))
            if temp2[0] - temp2[1] < coarsening:
                while coarsening > 0:
                    maxLevel = max(temp)
                    if maxLevel == self.lmin[0]:  # we assume here that lmin is equal everywhere
                        break
                    for d in range(self.dim):
                        if temp[d] == maxLevel:
                            temp[d] -= 1
                            coarsening -= 1
                            break
                area_is_null = True
            else:
                for d in range(self.dim):
                    if temp[d] == maxLevel:
                        temp[d] -= coarsening
                        break
                if area.is_already_calculated(tuple(temp), tuple(levelvector)):
                    area_is_null = True
                else:
                    area.add_level(tuple(temp), tuple(levelvector))
        else:
            while coarsening > 0:
                maxLevel = max(temp)
                if maxLevel == self.lmin[0]:  # we assume here that lmin is equal everywhere
                    break
                occurences_of_max = 0
                for i in temp:
                    if i == maxLevel:
                        occurences_of_max += 1
                is_top_diag = num_sub_diagonal == 0
                if self.version == 1:
                    no_forward_problem = coarsening_save >= self.lmax[0] + self.dim - 1 - maxLevel - (
                            self.dim - 2) - maxLevel + 1
                    do_coarsen = no_forward_problem and coarsening >= occurences_of_max - is_top_diag
                else:
                    no_forward_problem = coarsening_save >= self.lmax[0] + self.dim - 1 - maxLevel - (
                            self.dim - 2) - maxLevel + 2
                    do_coarsen = no_forward_problem and coarsening >= occurences_of_max
                if do_coarsen:
                    for d in range(self.dim):
                        if temp[d] == maxLevel:
                            temp[d] -= 1
                            coarsening -= 1
                else:
                    break
        level_coarse = [temp[d] - self.lmin[d] + int(self.noInitialSplitting) for d in range(len(temp))]
        return level_coarse, not area_is_null

    def initialize_refinement(self):
        if self.dim_adaptive:
            self.combischeme.init_adaptive_combi_scheme(self.lmax, self.lmin)
        if self.noInitialSplitting:
            assert False
            new_refinement_object = RefinementObjectExtendSplit(np.array(self.a), np.array(self.b), self.grid,
                                                                self.numberOfRefinementsBeforeExtend, 0, 0,
                                                                automatic_extend_split=self.automatic_extend_split)
            self.refinement = RefinementContainer([new_refinement_object], self.dim, self.errorEstimator)
        else:
            self.root_cell = RefinementObjectExtendSplit(np.array(self.a), np.array(self.b), self.grid,
                                                 self.numberOfRefinementsBeforeExtend, None, 0,
                                                 0, automatic_extend_split=self.automatic_extend_split)
            new_refinement_objects = self.root_cell.split_area_arbitrary_dim()
            self.refinement = RefinementContainer(new_refinement_objects, self.dim, self.errorEstimator)
            if self.operation is not None:
                #self.operation.area_preprocessing(parent)
                #self.compute_solutions([parent],[0])
                #self.operation.initialize_global_value(self.refinement)
                pass
                #self.operation.area_preprocessing(parent)
                #self.operation.evaluate_area(parent, np.zeros(self.dim, dtype=int), ComponentGridInfo(np.zeros(self.dim, dtype=int), 1), None, None)
                #self.operation.area_postprocessing(parent)
            else:
                parent_integral = self.grid.integrate(self.f, np.zeros(self.dim, dtype=int), self.a, self.b)
                self.root_cell.set_integral(parent_integral)
                self.refinement.integral = 0.0
        if self.errorEstimator is None:
            self.errorEstimator = ErrorCalculatorExtendSplit()

    def evaluate_area(self, f, area, component_grid, filter_area=None, interpolate=False):
        num_sub_diagonal = (self.lmax[0] + self.dim - 1) - np.sum(component_grid.levelvector)
        level_for_evaluation, do_compute = self.coarsen_grid(component_grid.levelvector, area, num_sub_diagonal)
        # print(level_for_evaluation, area.coarseningValue)
        if not do_compute:
            return None, None, 0
        else:
            if filter_area is None:
                return self.grid.integrate(f, level_for_evaluation, area.start, area.end), None, np.prod(
                    self.grid.levelToNumPoints(level_for_evaluation))
            else:
                if not interpolate:  # use filtering approach
                    self.grid.setCurrentArea(area.start, area.end, level_for_evaluation)
                    points, weights = self.grid.get_points_and_weights()
                    integral = 0.0
                    num_points = 0
                    for i, p in enumerate(points):
                        if self.point_in_area(p, filter_area):
                            integral += self.f(p) * weights[i] * self.get_point_factor(p, filter_area, area)
                            num_points += 1
                    return integral, None, num_points
                else:  # use bilinear interpolation to get function values in filter_area
                    integral = 0.0
                    num_points = 0
                    # create grid with boundaries; if 0 boudnary condition we will fill boundary points with 0's
                    boundary_save = self.grid.get_boundaries()
                    self.grid.set_boundaries([True] * self.dim)
                    self.grid.setCurrentArea(area.start, area.end, level_for_evaluation)
                    self.grid.set_boundaries(boundary_save)

                    # build grid consisting of corner points of area
                    corner_points = list(
                        zip(*[g.ravel() for g in
                              np.meshgrid(*[self.grid.coordinate_array[d] for d in range(self.dim)])]))

                    # calculate function values at corner points and transform  correct data structure for scipy
                    values = np.array([self.f(p) if self.grid.point_not_zero(p) else 0.0 for p in corner_points])
                    values = values.reshape(*[self.grid.numPointsWithBoundary[d] for d in reversed(range(self.dim))])
                    values = np.transpose(values)

                    # get corner grid in scipy data structure
                    corner_points_grid = [self.grid.coordinate_array[d] for d in range(self.dim)]

                    # get points of filter area for which we want interpolated values
                    self.grid.setCurrentArea(filter_area.start, filter_area.end, level_for_evaluation)
                    points, weights = self.grid.get_points_and_weights()

                    # bilinear interpolation
                    interpolated_values = interpn(corner_points_grid, values, points, method='linear')
                    # print(area.start, area.end, points,interpolated_values, weights)
                    integral += sum(
                        [interpolated_values[i] * weights[i] for i in range(len(interpolated_values))])
                    # print(corner_points)
                    for p in corner_points:
                        if self.point_in_area(p, filter_area) and self.grid.point_not_zero(p):
                            num_points += 1  # * self.get_point_factor(p,area,area_parent)
                    return integral, None, num_points

    def do_refinement(self, area, position):
        if self.automatic_extend_split:
            self.compute_benefits_for_operations(area)

        lmax_change = self.refinement.refine(position)
        if lmax_change != None:
            self.lmax = [self.lmax[d] + lmax_change[d] for d in range(self.dim)]
            if self.print_output:
                print("New scheme")
            self.scheme = self.combischeme.getCombiScheme(self.lmin[0], self.lmax[0],do_print=self.print_output)
            return True
        return False

    def compute_benefits_for_operations(self, area):
        # get integral values for the area for a potential parent that generated this area with an Extend and for
        # a potential parent that generated this area with a Split if necessary
        # in addition a reference is computed which is a Split + an Extend before the current refinement of area
        if self.operation is None:
            if area.parent_info.extend_parent_integral is None:
                area.parent_info.extend_parent_integral = self.get_parent_extend_integral(area)
            if area.parent_info.split_parent_integral is None and area.parent_info.benefit_extend is None:
                area.parent_info.split_parent_integral = self.get_parent_split_integral(area)
                self.get_reference_integral(area)
            self.set_extend_benefit(area)
            self.set_split_benefit(area)
            self.set_extend_error_correction(area)
        else:
            self.operation.initialize_error_estimates(area)
            if area.parent_info.benefit_split is None:
                self.get_parent_extend_operation(area)
            if area.parent_info.benefit_extend is None:
                self.get_parent_split_operation(area)
                self.get_reference_operation(area)
            self.operation.set_extend_benefit(area, self.norm)
            self.operation.set_split_benefit(area, self.norm)
            self.operation.set_extend_error_correction(area, self.norm)

    def set_extend_benefit(self, area):
        if area.parent_info.benefit_extend is not None:
            return
        if area.switch_to_parent_estimation:
            comparison = area.sum_siblings
            num_comparison = area.evaluations * 2 ** self.dim
        else:
            comparison = area.integral
            num_comparison = area.evaluations
        assert num_comparison > area.parent_info.num_points_extend_parent
        error_extend = LA.norm(abs((area.parent_info.split_parent_integral - comparison) / (abs(comparison) + 10 ** -100)), self.norm)
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
        assert num_comparison > area.parent_info.num_points_split_parent or area.switch_to_parent_estimation
        if self.grid.boundary:
            assert area.parent_info.num_points_split_parent > 0
        error_split = LA.norm(abs((area.parent_info.extend_parent_integral - area.integral) / (abs(area.integral) + 10 ** -100)), self.norm)
        if not self.grid.is_high_order_grid():
            area.parent_info.benefit_split = error_split * (area.parent_info.num_points_extend_parent - area.parent_info.num_points_reference)
        else:
            area.parent_info.benefit_split = error_split * area.parent_info.num_points_extend_parent

    def set_extend_error_correction(self, area):
        if area.switch_to_parent_estimation:
            area.parent_info.extend_error_correction = LA.norm(area.parent_info.extend_error_correction, self.norm) * area.parent_info.num_points_split_parent

    def calc_error(self, objectID, f):
        area = self.refinement.get_object(objectID)
        if area.parent_info.previous_value is None:
            if self.operation is None:
                integral2 = self.get_parent_split_integral(area, True)
                area.parent_info.previous_value = integral2
                area.parent_info.level_parent = self.lmax[0] - area.coarseningValue
                if area.switch_to_parent_estimation:
                    area.sum_siblings = 0.0
                    i = 0
                    for child in area.parent_info.parent.children:
                        if child.integral is not None:
                            area.sum_siblings += child.integral
                            i += 1
                    assert i == 2 ** self.dim  # we always have 2**dim children
            else:
                self.operation.initialize_error_estimates(area)
                self.get_parent_split_operation(area, True)
                self.operation.get_previous_value_from_split_parent(area)
                assert area.parent_info.previous_value is not None
                area.parent_info.level_parent = self.lmax[0] - area.coarseningValue
                if area.switch_to_parent_estimation:
                    self.operation.get_sum_sibling_value(area)
        else:
            area.sum_siblings = area.value if self.operation is not None else area.integral
        self.refinement.calc_error(objectID, f, self.norm)

    def get_parent_split_integral2(self, area, only_one_extend=False):
        area_parent = area.parent_info.parent

        if not area.switch_to_parent_estimation:

            coarsening = area.coarseningValue
            while True:

                parent_integral, num_points_split = self.evaluate_area_complete_flexibel(area_parent, coarsening,
                                                                                         filter_area=area,
                                                                                         filter_integral=True,
                                                                                         filter_points=True,
                                                                                         interpolate=False)
                area.parent_info.num_points_split_parent = num_points_split
                if only_one_extend or 3 * area.parent_info.num_points_split_parent > area.parent_info.num_points_extend_parent:
                    break
                else:
                    coarsening -= 1
        else:
            coarsening = area.coarseningValue

            while True:

                parent_integral, num_points_split = self.evaluate_area_complete_flexibel(area_parent, coarsening,
                                                                                         filter_area=area,
                                                                                         filter_integral=False,
                                                                                         filter_points=True,
                                                                                         interpolate=False)
                area.parent_info.num_points_split_parent = num_points_split
                if only_one_extend or 2 * area.parent_info.num_points_split_parent > area.parent_info.num_points_extend_parent:
                    break
                else:
                    coarsening -= 1

        return parent_integral

    def get_parent_split_integral(self, area, only_one_extend=False):
        area_parent = area.parent_info.parent
        if area.switch_to_parent_estimation:
            return self.get_parent_split_integral2(area, only_one_extend)
        else:
            coarsening = area.coarseningValue
            while True:
                parent_integral, num_points_split = self.evaluate_area_complete_flexibel(area_parent, coarsening,
                                                                                         filter_area=area,
                                                                                         filter_integral=True,
                                                                                         filter_points=True,
                                                                                         interpolate=True)

                area.parent_info.num_points_split_parent = num_points_split
                if only_one_extend or 3 * area.parent_info.num_points_split_parent > area.parent_info.num_points_extend_parent:
                    break
                else:
                    coarsening -= 1

        parent_integral2 = self.get_parent_split_integral2(area, only_one_extend)
        if abs(area.integral - parent_integral) < abs(area.integral - parent_integral2):
            return parent_integral
        else:
            return parent_integral2

    def get_parent_split_operation(self, area, only_one_extend=False):
        area_parent = area.parent_info.parent
        if not area.switch_to_parent_estimation:
            coarsening = area.coarseningValue
            while True:
                num_points_split = self.evaluate_operation_area_complete_flexibel(area_parent, coarsening,
                                                                                         filter_area=area,
                                                                                         filter_integral=True,
                                                                                         filter_points=True,
                                                                                         interpolate=True,
                                                                                  error_name="split_parent")

                area.parent_info.num_points_split_parent = num_points_split
                if only_one_extend or 3 * area.parent_info.num_points_split_parent > area.parent_info.num_points_extend_parent:
                    break
                else:
                    coarsening -= 1
        self.get_parent_split_operation2(area, only_one_extend)

    def get_parent_split_operation2(self, area, only_one_extend=False):
        area_parent = area.parent_info.parent

        if not area.switch_to_parent_estimation:

            coarsening = area.coarseningValue
            while True:

                num_points_split = self.evaluate_operation_area_complete_flexibel(area_parent, coarsening,
                                                                                         filter_area=area,
                                                                                         filter_integral=True,
                                                                                         filter_points=True,
                                                                                         interpolate=False,
                                                                                         error_name="split_parent2")
                area.parent_info.num_points_split_parent = num_points_split
                if only_one_extend or 3 * area.parent_info.num_points_split_parent > area.parent_info.num_points_extend_parent:
                    break
                else:
                    coarsening -= 1
        else:
            coarsening = area.coarseningValue

            while True:

                num_points_split = self.evaluate_operation_area_complete_flexibel(area_parent, coarsening,
                                                                                         filter_area=area,
                                                                                         filter_integral=False,
                                                                                         filter_points=True,
                                                                                         interpolate=False,
                                                                                  error_name="split_parent")
                area.parent_info.num_points_split_parent = num_points_split
                if only_one_extend or 2 * area.parent_info.num_points_split_parent > area.parent_info.num_points_extend_parent:
                    break
                else:
                    coarsening -= 1
        self.operation.get_best_fit(area, self.norm)

    def get_reference_integral(self, area):
        area_parent = area.parent_info.parent
        if area.switch_to_parent_estimation:
            reference_integral, num_points_reference = self.evaluate_area_complete_flexibel(area_parent,
                                                                                            area.coarseningValue + 1,
                                                                                            filter_area=area,
                                                                                            filter_integral=False,
                                                                                            filter_points=True,
                                                                                            interpolate=False)

        else:
            reference_integral, num_points_reference = self.evaluate_area_complete_flexibel(area_parent,
                                                                                            area.coarseningValue + 1,
                                                                                            filter_area=area,
                                                                                            filter_integral=True,
                                                                                            filter_points=True,
                                                                                            interpolate=False)
        area.parent_info.num_points_reference = num_points_reference
        return reference_integral

    def get_reference_operation(self, area):
        area_parent = area.parent_info.parent
        if area.switch_to_parent_estimation:
            num_points_reference = self.evaluate_operation_area_complete_flexibel(area_parent,
                                                                                            area.coarseningValue + 1,
                                                                                            filter_area=area,
                                                                                            filter_integral=False,
                                                                                            filter_points=True,
                                                                                            interpolate=False,
                                                                                  error_name="reference")

        else:
            num_points_reference = self.evaluate_operation_area_complete_flexibel(area_parent,
                                                                                            area.coarseningValue + 1,
                                                                                            filter_area=area,
                                                                                            filter_integral=True,
                                                                                            filter_points=True,
                                                                                            interpolate=False,
                                                                                  error_name="reference")
        area.parent_info.num_points_reference = num_points_reference

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

    def get_parent_extend_integral(self, area):

        if area.switch_to_parent_estimation:
            coarsening = self.lmax[0] - area.parent_info.level_parent if area.parent_info.level_parent != -1 else area.coarseningValue
            integral, num_points = self.evaluate_area_complete_flexibel(area, coarsening)
            area.parent_info.extend_error_correction = abs(area.integral - integral)

            extend_parent_integral, extend_num_points = self.evaluate_area_complete_flexibel(area, coarsening + 1)
            area.parent_info.num_points_extend_parent = extend_num_points

        else:
            extend_parent_integral, extend_num_points = self.evaluate_area_complete_flexibel(area,
                                                                                             area.coarseningValue + 1)
            area.parent_info.num_points_extend_parent = extend_num_points
        return extend_parent_integral

    def get_parent_extend_operation(self, area):

        if area.switch_to_parent_estimation:
            coarsening = self.lmax[0] - area.parent_info.level_parent if area.parent_info.level_parent != -1 else area.coarseningValue
            self.evaluate_operation_area_complete_flexibel(area, coarsening, error_name="extend_error_correction")
            #area.parent_info.extend_error_correction = abs(area.integral - integral)

            extend_num_points = self.evaluate_operation_area_complete_flexibel(area, coarsening + 1, error_name="extend_parent")
            area.parent_info.num_points_extend_parent = extend_num_points

        else:
            extend_num_points = self.evaluate_operation_area_complete_flexibel(area, area.coarseningValue + 1, error_name="extend_parent")
            area.parent_info.num_points_extend_parent = extend_num_points

    '''
    def get_parent_extend_integral2(self, area):
        lmax = self.lmax[0] - 1
        area.num_points_extend_parent = 0
        print(area.num_points_split_parent, area.split_parent_integral)
        while area.num_points_extend_parent <= area.num_points_split_parent:
            lmax += 1
            scheme = self.combischeme.getCombiScheme(self.lmin[0], lmax, do_print=False)
            if False: #area.switch_to_parent_estimation:
                area.num_points_extend_parent = 0.0
                extend_parent_integral = 0.0
                i = 0
                for area_eval in area.parent.children:
                    area_eval.levelvec_dict = {}
                    area_eval.coarseningValue += 1
                    i += 1
                    for ss in scheme:
                        if self.grid.isNested():
                            factor = ss[1]
                        else:
                            factor = 1
                        area_integral, partial_integrals, evaluations = self.evaluate_area(self.f, area_eval, ss[0])
                        #print(area_integral, partial_integrals, evaluations, area_eval.start, area_eval.end, ss[0])
                        area.num_points_extend_parent += evaluations * factor
                        extend_parent_integral += area_integral * ss[1]
                    area_eval.coarseningValue -= 1
                assert i == 2**self.dim
            else:
                area_eval = area
                extend_parent_integral = 0.0
                area_eval.levelvec_dict = {}
                area_eval.coarseningValue += 1
                area_eval.num_points_extend_parent = 0.0
                for ss in scheme:
                    if self.grid.isNested():
                        factor = ss[1]
                    else:
                        factor = 1
                    area_integral, partial_integrals, evaluations = self.evaluate_area(self.f, area_eval, ss[0])
                    #print(area_integral, partial_integrals, evaluations, area_eval.start, area_eval.end, ss[0])
                    area.num_points_extend_parent += evaluations * factor
                    extend_parent_integral += area_integral * ss[1]
                area_eval.coarseningValue -= 1
                print("Extend integral:", extend_parent_integral)
        return extend_parent_integral
    '''

    # This method computes the integral value for the combination technique with a specified coarsening
    # The level is calculated by lmax - coarsening.
    # It is possible to specify a filter area which defines a region for which we want an integral approximation based on the integral
    # for the area (the filtering area is smaller than the area)
    # In case we wilter we can specify
    def evaluate_area_complete_flexibel(self, area, coarsening, filter_area=None, filter_integral=False,
                                        filter_points=False, interpolate=False):
        integral = 0.0
        area.levelvec_dict = {}
        coarsening_save = area.coarseningValue
        area.coarseningValue = max(coarsening, 0)
        num_points = 0.0
        assert filter_integral == False or filter_points == True

        if coarsening >= 0:
            scheme = self.scheme
        else:
            lmax = self.lmax[0] + abs(coarsening)
            lmin = self.lmin[0]
            scheme = self.combischeme.getCombiScheme(lmin, lmax, do_print=False)

        for component_grid in scheme:
            if self.grid.isNested():
                factor = component_grid.coefficient
            else:
                factor = 1

            area_integral, partial_integrals, evaluations = self.evaluate_area(self.f, area, component_grid, filter_area,
                                                                               interpolate)
            if area_integral is not None:
                num_points += evaluations * factor
                integral += area_integral * component_grid.coefficient

        if not filter_integral and filter_points:
            integral = 0.0
            area.levelvec_dict = {}
            for component_grid in scheme:
                if area_integral is not None:
                    area_integral, partial_integrals, evaluations = self.evaluate_area(self.f, area, component_grid, None, None)
                    integral += area_integral * component_grid.coefficient

        area.coarseningValue = coarsening_save
        return integral, num_points


    # This method computes the integral value for the combination technique with a specified coarsening
    # The level is calculated by lmax - coarsening.
    # It is possible to specify a filter area which defines a region for which we want an integral approximation based on the integral
    # for the area (the filtering area is smaller than the area)
    # In case we wilter we can specify
    def evaluate_operation_area_complete_flexibel(self, area, coarsening, filter_area=None, filter_integral=False,
                                        filter_points=False, interpolate=False, error_name=None):
        area.levelvec_dict = {}
        coarsening_save = area.coarseningValue
        area.coarseningValue = max(coarsening, 0)
        num_points = 0.0
        assert filter_integral == False or filter_points == True

        if coarsening >= 0:
            scheme = self.scheme
        else:
            lmax = self.lmax[0] + abs(coarsening)
            lmin = self.lmin[0]
            scheme = self.combischeme.getCombiScheme(lmin, lmax, do_print=False)

        self.operation.initialize_error(filter_area if filter_area is not None else area, error_name)
        self.operation.initialize_point_numbers(area, error_name)

        additional_info = Operation_info(filter_area=filter_area, interpolate=interpolate, error_name=error_name)
        for component_grid in scheme:
            if self.grid.isNested() and self.operation.count_unique_points():
                factor = component_grid.coefficient
            else:
                factor = 1

            evaluations = self.evaluate_operation_area(component_grid, area, additional_info)
            num_points += evaluations * factor

        if not filter_integral and filter_points and error_name != "reference":
            area.levelvec_dict = {}
            self.operation.initialize_error(area, error_name)
            for component_grid in scheme:
                additional_info = Operation_info(target_area=filter_area, error_name="split_no_filter")
                evaluations = self.evaluate_operation_area(component_grid, area, additional_info)
        area.coarseningValue = coarsening_save
        return num_points

    def evaluate_operation_area(self, component_grid, area, additional_info=None):
        if additional_info is None:
            return super().evaluate_operation_area(component_grid, area)
        else:
            num_sub_diagonal = (self.lmax[0] + self.dim - 1) - np.sum(component_grid.levelvector)
            modified_levelvec, do_compute = self.coarsen_grid(component_grid.levelvector, area, num_sub_diagonal)
            if do_compute:
                evaluations = self.operation.evaluate_area_for_error_estimates(area, modified_levelvec, component_grid, self.refinement, additional_info)
                return evaluations
            else:
                return 0

class Operation_info(object):
    def __init__(self, filter_area=None, interpolate=False, error_name=None, target_area=None):
        self.filter_area = filter_area
        self.interpolate = interpolate
        self.error_name = error_name
        self.target_area = target_area
