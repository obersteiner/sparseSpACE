from sparseSpACE.spatiallyAdaptiveBase import *
from sparseSpACE.GridOperation import *

class SpatiallyAdaptiveExtendScheme(SpatiallyAdaptivBase):
    def __init__(self, a, b, number_of_refinements_before_extend=1, no_initial_splitting=False,
                 version=0, dim_adaptive=False, automatic_extend_split=False, split_single_dim=False, operation=None, norm=np.inf):
        # there are three different version that coarsen grids slightly different
        # version 0 coarsen as much as possible while extending and adding only new points in regions where it is supposed to
        # version 1 coarsens less and also adds moderately many points in non refined regions which might result in a more balanced configuration
        # version 2 coarsen fewest and adds a bit more points in non refinded regions but very similar to version 1
        assert 3 >= version >= 0
        self.version = version
        SpatiallyAdaptivBase.__init__(self, a=a, b=b, operation=operation, norm=norm)
        self.noInitialSplitting = no_initial_splitting
        self.numberOfRefinementsBeforeExtend = number_of_refinements_before_extend
        self.refinements_for_recalculate = 100
        self.dim_adaptive = dim_adaptive
        self.automatic_extend_split = automatic_extend_split
        self.split_single_dim = split_single_dim
        self.margin = 0.9

    def interpolate_points(self, interpolation_points, component_grid):
        point_assignements = self.get_points_assignement_to_areas(interpolation_points)
        dict_point_interpolation_values = {}
        f_value_array_length = self.operation.point_output_length()
        for area, contained_points in point_assignements:
            coarsened_levelvector, do_compute  = self.coarsen_grid(component_grid.levelvector, area)
            if do_compute:
                # check if dedicated interpolation routine is present in grid
                interpolation_op = getattr(self.grid, "interpolate", None)
                if callable(interpolation_op):
                    self.grid.setCurrentArea(start=area.start, end=area.end, levelvec=coarsened_levelvector)
                    interpolated_values = self.grid.interpolate(contained_points, area.start, area.end, coarsened_levelvector)
                else:
                    # call default d-linear interpolation based on points in grid
                    # Attention: This only works if we interpolate in between the grid points -> extrapolation not supported
                    self.grid.setCurrentArea(start=area.start, end=area.end, levelvec=coarsened_levelvector)
                    interpolated_values = self.operation.interpolate_points_component_grid(component_grid, self.grid.coordinate_array, contained_points)
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

    def add_refinment_to_figure_axe(self, ax, linewidth=1, fontsize=35):
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
            midpoint = 0.5*(np.asarray(i.start) + np.asarray(i.end))
            ax.text(midpoint[0] - 0*0.015, midpoint[1]-0.025, str(self.lmax[0] - i.coarseningValue),
                fontsize=fontsize, ha='center', color="blue")

    # returns the points of a single component grid with refinement
    def get_points_component_grid(self, levelvec):
        points_array = []
        for area in self.refinement.get_objects():
            start = area.start
            end = area.end
            level_interval, do_compute = self.coarsen_grid(levelvec, area)
            self.grid.setCurrentArea(start, end, level_interval)
            points = self.grid.getPoints()
            points_array.extend(points)
        return points_array

    def get_points_and_weights_component_grid(self, levelvec):
        points_array = []
        weights_array = []
        for area in self.refinement.get_objects():
            start = area.start
            end = area.end
            level_interval, do_compute = self.coarsen_grid(levelvec, area)
            self.grid.setCurrentArea(start, end, level_interval)
            points, weights = self.grid.get_points_and_weights()
            points_array.extend(points)
            weights_array.extend(weights)
        return points_array, weights_array

    # returns the points of a single component grid with refinement
    def get_points_component_grid_not_null(self, levelvec):
        array2 = []
        for area in self.refinement.get_objects():
            start = area.start
            end = area.end
            level_interval, do_compute = self.coarsen_grid(levelvec, area)
            if do_compute:
                self.grid.setCurrentArea(start, end, level_interval)
                points = self.grid.getPoints()
                array2.extend(points)
                # print("considered", levelvec, level_interval, area.start, area.end, area.coarseningValue)
            # else:
            # print("not considered", levelvec, level_interval, area.start, area.end, area.coarseningValue)
        return array2

    # optimized adaptive refinement refine multiple cells in close range around max variance (here set to 10%)
    def coarsen_grid(self, levelvector, area):
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
        elif self.version == 3:
            num_sub_diagonal = (self.lmax[0] + self.dim - 1) - np.sum(levelvector)
            assert (num_sub_diagonal < self.dim)
            currentDirection = 0
            num_sub_diagonal_save = num_sub_diagonal
            while coarsening > 0:
                if temp[currentDirection] > self.lmin[currentDirection]:
                    temp[currentDirection] -= 1
                #    coarsening -= 1
                #else:
                #    if num_sub_diagonal_save > 0:
                #        coarsening -= 1
                #        num_sub_diagonal_save -= 1
                coarsening -= 1
                currentDirection = (currentDirection + 1) % self.dim
        else:
            num_sub_diagonal = (self.lmax[0] + self.dim - 1) - np.sum(levelvector)
            assert (num_sub_diagonal < self.dim)
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
                                                                self.numberOfRefinementsBeforeExtend,
                                                                automatic_extend_split=self.automatic_extend_split,
                                                                splitSingleDim=self.split_single_dim)
            self.refinement = RefinementContainer([new_refinement_object], self.dim, self.errorEstimator)
        else:
            self.root_cell = RefinementObjectExtendSplit(np.array(self.a), np.array(self.b), self.grid,
                                                 self.numberOfRefinementsBeforeExtend, None, 0,
                                                 0, automatic_extend_split=self.automatic_extend_split,
                                                 splitSingleDim=self.split_single_dim)
            if self.split_single_dim:
                self.root_cell.numberOfRefinementsBeforeExtend += self.dim
                new_refinement_objects = [self.root_cell]
                for d in range(self.dim):
                    temp = []
                    for area in new_refinement_objects:
                        temp.extend(area.split_area_single_dim(d))
                    new_refinement_objects = temp
                assert len(new_refinement_objects) == 2**self.dim
                for area in new_refinement_objects:
                    for d in range(self.dim):
                        area.twins[d] = None
                for i in range(2**self.dim):
                    area = new_refinement_objects[i]
                    for d in range(self.dim):
                        '''
                        twin = new_refinement_objects[(i+2**(self.dim-1)) % 2**(self.dim-d)]
                        area.set_twin(d, twin)
                        if area.twinErrors[d] is None:
                            area.set_twin_error(d, abs(area.integral - twin.integral))
                        print("Area", area.start, area.end, "has twin", twin.start, twin.end, "in dimension", d)
                        '''
                        if area.twins[d] is None:
                            twin_distance = 2**(self.dim - d - 1)
                            twin = new_refinement_objects[i + twin_distance]
                            area.set_twin(d, twin)
                            #if area.twinErrors[d] is None:
                            #    area.set_twin_error(d, abs(area.integral - twin.integral))
                            #print("Area", area.start, area.end, "has twin", twin.start, twin.end, "in dimension", d)
                    #if area.twinErrors[self.dim-1] is None:
                    #    area.set_twin_error(self.dim-1, abs(area.integral - area.twins[self.dim-1].integral))
                    area.parent_info.parent = self.root_cell
                self.calculate_new_twin_errors(new_refinement_objects)
                for area in new_refinement_objects:
                    area.parent_info.parent = self.root_cell
                self.root_cell.children = new_refinement_objects
            else:
                self.root_cell.numberOfRefinementsBeforeExtend += 1
                new_refinement_objects = self.root_cell.split_area_arbitrary_dim()
            self.refinement = RefinementContainer(new_refinement_objects, self.dim, self.errorEstimator)
        if self.errorEstimator is None:
            self.errorEstimator = ErrorCalculatorExtendSplit()

    def calculate_new_twin_errors(self, new_refinement_objects):
        for area in new_refinement_objects:
            for component_grid in self.scheme:
                modified_levelvec, do_compute = self.coarsen_grid(component_grid.levelvector, area)
                if do_compute:
                    evaluations = self.operation.evaluate_area(area, modified_levelvec, component_grid, None, None, apply_to_combi_result=False)

        for area in new_refinement_objects:
            #print("area", area.start,area.end, area.integral)
            for d in range(self.dim):
                ##print("Current dim:", d)
                if area.twinErrors[d] is None:
                    assert area.twins[d] is not None
                    parent_region_a = np.array(area.start)
                    parent_region_b = np.array(area.end)
                    parent_region_a[d] = min(area.start[d], area.twins[d].start[d])
                    parent_region_b[d] = max(area.end[d], area.twins[d].end[d])
                    parent_integral = 0.0
                    parent_area = RefinementObjectExtendSplit(parent_region_a, parent_region_b, self.grid,
                                                        self.numberOfRefinementsBeforeExtend, None, area.coarseningValue, area.needExtendScheme,
                                                        automatic_extend_split=self.automatic_extend_split,
                                                        splitSingleDim=self.split_single_dim)
                    area.parent_info.parent = parent_area
                    for component_grid in self.scheme:
                        modified_levelvec, do_compute = self.coarsen_grid(component_grid.levelvector, parent_area)
                        if do_compute:
                            evaluations = self.operation.evaluate_area(parent_area, modified_levelvec, component_grid, None, None, apply_to_combi_result=False)
                    #print("Areas", area.start, area.end, area.twins[d].start, area.twins[d].end)
                    #print("Integrals", area.integral, area.twins[d].integral, area.parent_info.parent.integral)
                    twin_error = self.get_twin_error(d, area, self.norm)
                    area.set_twin_error(d, twin_error)

    def do_refinement(self, area, position):
        if self.automatic_extend_split:
            self.compute_benefits_for_operations(area)
        if self.split_single_dim:
            for d in range(self.dim):
                if area.twinErrors[d] is None:
                    twinError = self.get_twin_error(d, area,self.norm)
                    area.set_twin_error(d, twinError)
        lmax_change, new_objects = self.refinement.refine(position)
        if self.split_single_dim and len(new_objects) > 2:
           self.calculate_new_twin_errors(new_objects)

        if lmax_change != None:
            self.lmax = [self.lmax[d] + lmax_change[d] for d in range(self.dim)]
            if self.print_output:
                print("New scheme")
            self.scheme = self.combischeme.getCombiScheme(self.lmin[0], self.lmax[0],do_print=self.print_output)
            return False
        return False

    def compute_benefits_for_operations(self, area):
        # get integral values for the area for a potential parent that generated this area with an Extend and for
        # a potential parent that generated this area with a Split if necessary
        # in addition a reference is computed which is a Split + an Extend before the current refinement of area
        self.initialize_error_estimates(area)
        if area.parent_info.benefit_split is None:
            self.get_parent_extend_operation(area)
        if area.parent_info.benefit_extend is None:
            self.get_parent_split_operation(area)
            self.get_reference_operation(area)
        self.set_extend_benefit(area, self.norm)
        self.set_split_benefit(area, self.norm)
        self.set_extend_error_correction(area, self.norm)

    def calc_error(self, objectID):
        area = self.refinement.get_object(objectID)
        if area.parent_info.previous_value is None:
            self.initialize_error_estimates(area)
            self.get_parent_split_operation(area, True)
            self.get_previous_value_from_split_parent(area)
            assert area.parent_info.previous_value is not None
            area.parent_info.level_parent = self.lmax[0] - area.coarseningValue
            if area.switch_to_parent_estimation:
                self.get_sum_sibling_value(area)
        else:
            area.sum_siblings = area.value if self.operation is not None else area.value
        self.refinement.calc_error(objectID, self.norm)

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
        self.get_best_fit(area, self.norm)

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

        self.initialize_error(filter_area if filter_area is not None else area, error_name)
        self.initialize_point_numbers(area, error_name)

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
            self.initialize_error(filter_area, error_name)
            for component_grid in scheme:
                additional_info = Operation_info(target_area=filter_area, error_name="split_no_filter")
                evaluations = self.evaluate_operation_area(component_grid, area, additional_info)
        area.coarseningValue = coarsening_save
        return num_points

    def evaluate_operation_area(self, component_grid, area, additional_info=None):
        if additional_info is None:
            return super().evaluate_operation_area(component_grid, area)
        else:
            modified_levelvec, do_compute = self.coarsen_grid(component_grid.levelvector, area)
            if do_compute:
                evaluations = self.operation.evaluate_area_for_error_estimates(area, modified_levelvec, component_grid, self.refinement, additional_info)
                return evaluations
            else:
                return 0

    def initialize_error_estimates(self, area):
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

    def get_best_fit(self, area, norm):
        old_value = area.parent_info.split_parent_integral
        new_value = area.parent_info.split_parent_integral2
        if old_value is None:
            area.parent_info.split_parent_integral = np.array(new_value)
        else:
            if new_value is None or self.operation.compute_difference(area.value, old_value, norm=norm) < self.operation.compute_difference(area.value, new_value, norm=norm):
                pass
            else:
                area.parent_info.split_parent_integral = np.array(area.parent_info.split_parent_integral2)

    def get_previous_value_from_split_parent(self, area):
        area.parent_info.previous_value = area.parent_info.split_parent_integral

    def get_twin_error(self, d, area, norm):
        return self.operation.compute_difference(area.parent_info.parent.value, self.operation.add_values(area.value, area.twins[d].value), norm)

    def set_extend_benefit(self, area, norm):
        if area.parent_info.benefit_extend is not None:
            return
        if area.switch_to_parent_estimation:
            comparison = area.sum_siblings
            num_comparison = area.evaluations * 2 ** self.dim
        else:
            comparison = area.value
            num_comparison = area.evaluations
        assert num_comparison > area.parent_info.num_points_split_parent or area.switch_to_parent_estimation

        error_extend = self.operation.compute_difference(area.parent_info.split_parent_integral / (abs(comparison) + 10 ** -100), comparison / (abs(comparison) + 10 ** -100) , norm)
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
        error_split = self.operation.compute_difference(area.parent_info.extend_parent_integral / (abs(area.value) + 10 ** -100), area.value / (abs(area.value) + 10 ** -100), norm)
        if not self.grid.is_high_order_grid():
            area.parent_info.benefit_split = error_split * (area.parent_info.num_points_extend_parent - area.parent_info.num_points_reference)
        else:
            area.parent_info.benefit_split = error_split * area.parent_info.num_points_extend_parent

    def set_extend_error_correction(self, area, norm):
        if area.switch_to_parent_estimation:
            area.parent_info.extend_error_correction = LA.norm(area.parent_info.extend_error_correction, norm) * area.parent_info.num_points_split_parent

    def get_sum_sibling_value(self, area):
        area.sum_siblings = np.zeros(self.operation.point_output_length())
        i = 0
        for child in area.parent_info.parent.children:
            if child.value is not None:
                area.sum_siblings = self.operation.add_values(child.value, area.sum_siblings)
                i += 1
        assert i == 2 ** self.dim or i == 2  # we always have 2**dim children

class Operation_info(object):
    def __init__(self, filter_area=None, interpolate=False, error_name=None, target_area=None):
        self.filter_area = filter_area
        self.interpolate = interpolate
        self.error_name = error_name
        self.target_area = target_area
