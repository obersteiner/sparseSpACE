from spatiallyAdaptiveBase import *
from GridOperation import *

def sortToRefinePosition(elem):
    # sort by depth
    return elem[1]


class SpatiallyAdaptiveSingleDimensions2(SpatiallyAdaptivBase):
    def __init__(self, a: Sequence[float], b: Sequence[float], norm: int=np.inf, dim_adaptive: bool=True, version: int=3, operation: GridOperation=None, margin: float=None, rebalancing: bool=True, chebyshev_points=False):
        SpatiallyAdaptivBase.__init__(self, a, b, operation=operation, norm=norm)
        assert self.grid is not None
        self.grid_surplusses = self.grid #GlobalTrapezoidalGrid(a, b, boundary=boundary, modified_basis=modified_basis)
        self.dim_adaptive = dim_adaptive
        #self.evaluationCounts = None
        self.version = version
        self.dict_integral = {}
        self.dict_points = {}
        self.no_previous_integrals = True
        self.use_local_children = True #self.version == 2 or self.version == 3
        if margin is None:
            self.margin = 0.9#1-10**-12 if self.use_local_children else 0.9
        else:
            self.margin = margin
        self.operation = operation
        self.equidistant = False #True
        self.rebalancing = rebalancing
        self.subtraction_value_cache = {}
        self.max_level_dict = {}
        self.chebyshev_points = chebyshev_points


    def interpolate_points(self, interpolation_points: Sequence[Tuple[float, ...]], component_grid: ComponentGridInfo) -> Sequence[Sequence[float]]:
        # check if dedicated interpolation routine is present in grid
        interpolation_op = getattr(self.grid, "interpolate", None)
        if callable(interpolation_op):
            gridPointCoordsAsStripes, grid_point_levels, children_indices = self.get_point_coord_for_each_dim(component_grid.levelvector)
            self.grid.set_grid(gridPointCoordsAsStripes, grid_point_levels)
            return self.grid.interpolate(interpolation_points, component_grid)
        else:
            # call default d-linear interpolation based on points in grid
            # Attention: This only works if we interpolate in between the grid points -> extrapolation not supported
            gridPointCoordsAsStripes, grid_point_levels, children_indices = self.get_point_coord_for_each_dim(component_grid.levelvector)
            return self.operation.interpolate_points(self.operation.get_component_grid_values(component_grid, gridPointCoordsAsStripes), gridPointCoordsAsStripes, interpolation_points)

    def interpolate_grid_component(self, grid_coordinates: Sequence[Sequence[float]], component_grid: ComponentGridInfo) -> Sequence[Sequence[float]]:
        # check if dedicated interpolation routine is present in grid
        interpolation_op = getattr(self.grid, "interpolate_grid", None)
        if callable(interpolation_op):
            gridPointCoordsAsStripes, grid_point_levels, children_indices = self.get_point_coord_for_each_dim(component_grid.levelvector)
            self.grid.set_grid(gridPointCoordsAsStripes, grid_point_levels)
            return self.grid.interpolate_grid(grid_coordinates, component_grid)
        else:
            # call default d-linear interpolation based on points in grid
            # Attention: This only works if we interpolate in between the grid points -> extrapolation not supported
            return super().interpolate_grid_component(grid_coordinates, component_grid)

    def coarsen_grid(self, area, levelvec: Sequence[int]):
        pass

    # returns the points coordinates of a single component grid with refinement
    def get_points_all_dim(self, levelvec: Sequence[int]) -> List[Tuple[float, ...]]:
        indicesList, grid_point_levels, children_indices = self.get_point_coord_for_each_dim(levelvec)
        if not self.grid.boundary:
            indicesList = [indices[1:-1] for indices in indicesList]
        # this command creates tuples of size this_dim of all combinations of indices (e.g. this_dim = 2 indices = ([0,1],[0,1,2,3]) -> areas = [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3)] )
        allPoints = list(set(zip(*[g.ravel() for g in np.meshgrid(*indicesList)])))
        return allPoints

    # returns the points of a single component grid with refinement
    def get_points_component_grid(self, levelvec: Sequence[int]) -> List[Tuple[float, ...]]:
        return self.get_points_all_dim(levelvec)

    def get_points_and_weights_component_grid(self, levelvec: Sequence[int]) -> Tuple[Sequence[Tuple[float, ...]], Sequence[float]]:
        point_coords, point_levels, _ =self.get_point_coord_for_each_dim(levelvec)
        self.grid.set_grid(point_coords, point_levels)
        points, weights = self.grid.get_points_and_weights()
        return points, weights

    def get_num_points_each_dim(self):
        num_points = np.zeros(self.dim, dtype=int)
        for component_grid in self.scheme:
            point_coords, point_levels, _ = self.get_point_coord_for_each_dim(component_grid.levelvector)
            self.grid.set_grid(point_coords, point_levels)
            num_points_component_grid = self.grid.levelToNumPoints(component_grid.levelvector)
            for i, v in enumerate(num_points_component_grid):
                if num_points[i] < v:
                    num_points[i] = v
        assert all([v > 0 for v in num_points])
        return num_points

    # returns list of coordinates for each dimension (basically refinement stripes) + all points that are associated
    # with a child in the global refinement structure. There might be now such points that correspond to a global child.
    def get_point_coord_for_each_dim(self, levelvec: Sequence[int]):
        refinement = self.refinement
        # get a list of all coordinates for every this_dim (so (0, 1), (0, 0.5, 1) for example)
        indicesList = []
        children_indices = []
        indices_level = []
        max_coarsenings = np.zeros(self.dim, dtype=int) #only for dimensions with level > 1
        for d in range(self.dim):
            #if levelvec[d] > self.lmin[d]:
            max_coarsenings[d] = refinement.get_max_coarsening(d)
        for d in range(0, self.dim):
            max_coarsenings_dim = list(max_coarsenings)
            refineContainer = refinement.get_refinement_container_for_dim(d)
            refine_container_objects = refineContainer.get_objects()
            indicesDim = []
            indices_levelDim = []

            children_indices_dim = []
            indicesDim.append(refine_container_objects[0].start)
            indices_levelDim.append(refine_container_objects[0].levels[0])
            for i in range(len(refine_container_objects)):
                refineObj = refine_container_objects[i]
                if i + 1 < len(refine_container_objects):
                    next_refineObj = refine_container_objects[i + 1]
                else:
                    next_refineObj = None

                subtraction_value = self.get_subtraction_value(refineObj, refineContainer, i, max_coarsenings, d, levelvec)

                if (refineObj.levels[1] <= max(levelvec[d] - subtraction_value, 1)):

                    indicesDim.append(refineObj.end)
                    if not self.use_local_children:
                        if next_refineObj is not None and self.is_child(refineObj.levels[0], refineObj.levels[1], next_refineObj.levels[0]):
                            children_indices_dim.append(self.get_node_info(refineObj.end, refineObj.levels[1], refineObj.start, refineObj.levels[0], next_refineObj.end, next_refineObj.levels[1], d))
                    else:
                        indices_levelDim.append(refineObj.levels[1])
                        #print(d, refineObj.end)

            if self.use_local_children:
                for i in range(1,len(indices_levelDim)-1):
                    if self.is_child(indices_levelDim[i-1], indices_levelDim[i], indices_levelDim[i+1]):
                        #print(d, indicesDim[i])
                        children_indices_dim.append((self.get_node_info(i, indicesDim, indices_levelDim, d)))
                        #print(children_indices_dim, d)
            indicesList.append(indicesDim)
            children_indices.append(children_indices_dim)
            indices_level.append(indices_levelDim)

            # Test if children_indices is valid
            for c in children_indices_dim:
                indicesDim.index(c.left_parent)
                indicesDim.index(c.right_parent)
            # Test if indices are valid
            assert all(indicesDim[i] <= indicesDim[i + 1] for i in range(len(indicesDim) - 1))

        return indicesList, indices_level, children_indices


    def modify_according_to_levelvec(self, subtraction_value, d, max_level, levelvec):
        if levelvec[d] - subtraction_value == max_level and levelvec[d] < self.lmax[d]:
            subtraction_value += 1
        subtraction_value = min(subtraction_value, levelvec[d] - self.lmin[d])
        return subtraction_value

    def get_subtraction_value(self, refineObj, refineContainer, i, max_coarsenings, d, levelvec):
        if self.version == 5:
            if tuple((d, i)) in self.max_level_dict:
                if tuple((d, self.max_level_dict[tuple((d, i))])) in self.subtraction_value_cache:
                    subtraction_value = self.subtraction_value_cache[tuple((d, self.max_level_dict[tuple((d, i))]))]
                    return self.modify_according_to_levelvec(subtraction_value, d, self.max_level_dict[tuple((d, i))],
                                                             levelvec)
            #if self.combischeme.has_forward_neighbour(levelvec):
                #subtraction_value = None
                #for k in range(self.dim):
                #    if k != d:
                #        levelvec_temp = list(levelvec)
                #        levelvec_temp[k] += 1
                #        if self.combischeme.in_index_set(levelvec_temp):
                #            subtract_value_temp = self.get_subtraction_value(refineObj, refineContainer, i,
                #                                              max_coarsenings, d, levelvec_temp)
                #            #print(subtraction_value, subtract_value_temp, levelvec, d)
                #            if subtraction_value is not None:
                #                assert subtract_value_temp == subtraction_value
                #            else:
                #                subtraction_value = subtract_value_temp
                #return self.modify_according_to_levelvec(subtraction_value, d, self.max_level_dict[tuple((d,i))], levelvec)
        if self.version == 4 or self.version == 5:
            #max_coarsenings_levelvec = [coarsening if levelvec[k] > 1 else 0 for k, coarsening in enumerate(max_coarsenings)]
            max_coarsenings_levelvec = max_coarsenings
        if self.version == 2 or self.version == 3 or self.version == 4 or self.version == 5:
            refineObj_temp = refineObj
            if not tuple((d,i)) in self.max_level_dict:
                max_level = refineObj_temp.levels[1]
                k = 0
                while (i - k > 0):
                    refineObj_temp = refineContainer.get_objects()[i - k]
                    max_level = max(max_level, refineObj_temp.levels[0])
                    if refineObj_temp.levels[0] <= refineObj.levels[1]:
                        break
                    k += 1
                k = 1
                while (i + k < len(refineContainer.get_objects())):
                    refineObj_temp = refineContainer.get_objects()[i + k]
                    max_level = max(max_level, refineObj_temp.levels[1])
                    if refineObj_temp.levels[1] <= refineObj.levels[1]:
                        break
                    k += 1
            else:
                max_level = self.max_level_dict[tuple((d,i))]
            subtraction_value = (self.lmax[d] - max_level)
            self.max_level_dict[tuple((d,i))] = max_level
            if (self.version == 4 or self.version == 5) and max_level > 2:
                # if levelvec[d] > self.lmax[d] - subtraction_value:
                #    subtraction_value = self.lmax[d]
                # print(max_coarsenings)
                max_coarsening_other_dims = max([max_coarsenings_levelvec[i] if i != d else 0 for i in range(self.dim)])
                if self.version == 5:
                    max_coarsening_other_dims = min(subtraction_value, max_coarsening_other_dims)
                if max_coarsening_other_dims != 0 and tuple((d, max_level)) not in self.subtraction_value_cache:
                    if max_coarsening_other_dims < subtraction_value:
                        subtraction_value_new = subtraction_value - max_coarsening_other_dims
                        max_coarsenings_temp = list(max_coarsenings_levelvec)
                        max_coarsenings_temp[d] = subtraction_value
                        # if self.version == 5:
                        #    max_coarsenings_temp = [coarsening if coarsening < subtraction_value else subtraction_value for coarsening in max_coarsenings_temp]
                        max_coarsenings_temp[d] -= subtraction_value_new
                        assert subtraction_value >= subtraction_value_new
                        remainder = subtraction_value - subtraction_value_new
                    else:
                        max_coarsenings_temp = [subtraction_value if coarsening > subtraction_value else coarsening for
                                                coarsening in max_coarsenings_levelvec]
                        if self.version == 5:
                            remainder = subtraction_value
                        else:
                            remainder = subtraction_value - sum(
                                np.asarray(max_coarsenings_levelvec) - np.asarray(max_coarsenings_temp)) + max_coarsenings_levelvec[
                                            d] - subtraction_value
                        if self.version == 5:
                            # print(max_coarsenings_temp, max_coarsenings, remainder, sum(np.asarray(max_coarsenings) - np.asarray(max_coarsenings_temp)))
                            assert remainder == subtraction_value
                        subtraction_value_new = 0
                    while (remainder > 0):
                        #print(max_coarsenings_temp, remainder)
                        max_coarsening = max(max_coarsenings_temp)
                        second_largest_coarsening = max(
                            [coarsening if coarsening < max_coarsening else 0 for coarsening in max_coarsenings_temp])
                        assert max_coarsening > 0
                        # print(remainder, max_coarsening, second_largest_coarsening, max_coarsenings_temp)
                        # print(max_coarsening, second_largest_coarsening, max_coarsenings_temp)
                        max_dimensions = [1 if coarsening == max_coarsening else 0 for coarsening in
                                          max_coarsenings_temp]
                        num_max_coarsening = min(sum(max_dimensions), min((max_level - 1), sum([1 if self.lmax[k] > 2 else 0 for k in range(self.dim)])))
                        #print(num_max_coarsening, self.combischeme.lmax_adaptive - max(sum([1 if self.lmax[k] <= 2 else 0 for k in range(self.dim)]), 1), self.combischeme.lmax_adaptive, self.lmax)
                        my_position = sum(max_dimensions[:d])
                        if remainder >= (max_coarsening - second_largest_coarsening) * num_max_coarsening:
                            subtraction_value_new += max_coarsening - second_largest_coarsening
                            remainder -= (max_coarsening - second_largest_coarsening) * num_max_coarsening
                        else:

                            added_subtraction_value = int(remainder / num_max_coarsening)
                            #subtraction_value_new += added_subtraction_value
                            if added_subtraction_value - int(
                                    added_subtraction_value) > my_position / num_max_coarsening:
                                subtraction_value_new += int(math.ceil(added_subtraction_value))
                            else:
                                subtraction_value_new += int(added_subtraction_value)
                            break
                        max_coarsenings_temp = [coarsening if coarsening < max_coarsening else second_largest_coarsening
                                                for coarsening in max_coarsenings_temp]
                        assert subtraction_value >= subtraction_value_new
                    assert subtraction_value >= subtraction_value_new
                    subtraction_value = subtraction_value_new
                    # print(subtraction_value, d, max_level, max_coarsenings)
                    self.subtraction_value_cache[tuple((d, max_level))] = subtraction_value
                else:
                    if tuple((d, max_level)) in self.subtraction_value_cache:
                        subtraction_value = self.subtraction_value_cache[tuple((d, max_level))]
                subtraction_value = self.modify_according_to_levelvec(subtraction_value, d, self.max_level_dict[tuple((d, i))], levelvec)
                # print(subtraction_value, d, max_level, max_coarsenings)
            if self.version == 3 and max_level > 2:
                subtraction_value /= self.dim
                if subtraction_value - int(subtraction_value) > d / self.dim:
                    subtraction_value = int(math.ceil(subtraction_value))
                else:
                    subtraction_value = int(subtraction_value)
            # subtraction_value = max(refineObj.coarsening_level, next_refineObj.coarsening_level) if next_refineObj is not None else refineObj.coarsening_level

        else:
            subtraction_value = 0
        return subtraction_value


    # returns if the coordinate refineObj.levels[1] is a child in the global refinement structure level 1 is never considered to be a child
    def is_child(self, level_left_point: int, level_point: int, level_right_point: int) -> bool:
        return (level_left_point < level_point and level_right_point < level_point) and level_point > 1
        #return True
        #if level_left_point < level_point or level_right_point < level_point:
        #    return True
        #else:
        #    return False

    # This method calculates the left and right parent of a child. It might happen that a child has already a child
    # in one direction but it may not have one in both as it would not be considered to be a child anymore.
    def get_node_info(self, position, coords_dim, level_dim, d):
        child = coords_dim[position]
        level_child = level_dim[position]
        if self.equidistant:
            width = (self.b[d] - self.a[d]) / 2**level_child
            right_parent_of_right_parent = child + 2* width if level_child > 1 else None
            left_parent_of_left_parent = child - 2 * width if level_child > 1 else None
            return NodeInfo(child, child - width, child + width, left_parent_of_left_parent , right_parent_of_right_parent ,True, True, None,None, level_child)
        else:
            left_parent = None
            level_parent = None
            left_of_left_parent = None
            for i in reversed(range(position)):
                if level_dim[i] < level_child and left_parent is None:
                    left_parent = coords_dim[i]
                    level_parent = level_dim[i]
                if left_parent is not None and level_dim[i] < level_parent:
                    left_of_left_parent = coords_dim[i]
                    break

            assert left_parent is not None
            right_parent = None
            level_parent = None
            right_of_right_parent = None
            for i in (range(position+1, len(coords_dim))):
                if level_dim[i] < level_child and right_parent is None:
                    right_parent = coords_dim[i]
                    level_parent = level_dim[i]
                if right_parent is not None and level_dim[i] < level_parent:
                    right_of_right_parent = coords_dim[i]
                    break
            assert right_parent is not None
            return NodeInfo(child, left_parent, right_parent, left_of_left_parent, right_of_right_parent, True, True, None,None, level_child)

    # this method draws the 1D refinement of each dimension individually
    def draw_refinement(self, filename: str=None, markersize:int =20, fontsize=60, single_dim:int=None):  # update with meta container
        plt.rcParams.update({'font.size': fontsize})
        refinement = self.refinement
        dim = self.dim if single_dim is None else 1
        fig, ax = plt.subplots(ncols=1, nrows=dim, figsize=(20, 5*dim))
        offset = 0 if single_dim is None else single_dim
        for d in range(dim):
            axis = ax[d] if single_dim is None else ax
            objs = refinement.refinementContainers[d+offset].get_objects()
            infinite_bounds = isinf(objs[0].start)
            if infinite_bounds:
                # Ignore refinement objects with infinite borders
                objs = objs[1:-1]
            starts = [refinementObject.start for refinementObject in objs]
            starts_levels = [refinementObject.levels[0] for refinementObject in objs]
            ends = [refinementObject.end for refinementObject in objs]
            ends_levels = [refinementObject.levels[1] for refinementObject in objs]
            for i in range(len(starts)):
                axis.add_patch(
                    patches.Rectangle(
                        (starts[i], -0.1),
                        ends[i] - starts[i],
                        0.2, linestyle='-',
                        fill=False  # remove background
                    )
                )
                axis.text(starts[i]+0.015, 0.01, str(starts_levels[i]),
                          fontsize=fontsize-10, ha='center', color="blue")
            axis.text(ends[-1] - 0.015, 0.01, str(ends_levels[-1]),
                fontsize=fontsize-10, ha='center', color="blue")
            xValues = starts + ends
            yValues = np.zeros(len(xValues))
            axis.plot(xValues, yValues, 'bo', markersize=markersize, color="black")
            if infinite_bounds:
                start, end = objs[0].start, objs[-1].end
                # offset_bound = (end - start) * 0.1
            else:
                start, end = self.a[d], self.b[d]
                # offset_bound = (end - start) * 0.005
            axis.set_xlim([start-0.005, end+0.005])
            axis.set_ylim([-0.05, 0.05])
            axis.set_yticks([])
            axis.set_title("$x_" + str(d + 1 + offset) + "$")

        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
        plt.show()
        return fig

    # this method draws the 1D refinement of each dimension individually
    def draw_refinement_trees(self, filename: str=None, markersize:int =20, fontsize=20, single_dim:int=None):  # update with meta container
        plt.rcParams.update({'font.size': 60})
        refinement = self.refinement
        dim = self.dim if single_dim is None else 1
        height = 5*sum(self.lmax) if single_dim is None else 5 * self.lmax[single_dim]
        fig, ax = plt.subplots(ncols=1, nrows=dim if single_dim is None else 1, figsize=(20, height))
        offset = 0 if single_dim is None else single_dim
        for d in range(dim):
            axis = ax[d] if single_dim is None else ax
            starts = [refinementObject.start for refinementObject in refinement.refinementContainers[d + offset].get_objects()]
            starts_levels = [refinementObject.levels[0] for refinementObject in refinement.refinementContainers[d + offset].get_objects()]
            ends = [refinementObject.end for refinementObject in refinement.refinementContainers[d + offset].get_objects()]
            ends_levels = [refinementObject.levels[1] for refinementObject in refinement.refinementContainers[d + offset].get_objects()]
            max_level = max(starts_levels)
            yvalues = np.zeros(len(starts) + 1)
            for i in range(len(starts)):
                x_position = starts[i]
                y_position = starts_levels[i]
                yvalues[i] = y_position
                if i !=0:
                    for j in reversed(range(i)):
                        if starts_levels[j] == starts_levels[i] + 1:
                            target_x_position = starts[j]
                            target_y_position = starts_levels[j]
                            axis.arrow(x_position, y_position, target_x_position - x_position + 0.001,
                                        target_y_position - y_position - 0.04,
                                        head_width=0.04, head_length=0.04, fc='k', ec='k', length_includes_head=True, overhang=0, capstyle="butt")
                            break
                        if starts_levels[j] <= starts_levels[i]:
                            break

                for j in range(i+1, len(starts)):
                    if starts_levels[j] == starts_levels[i] + 1:
                        target_x_position = starts[j]
                        target_y_position = starts_levels[j]
                        axis.arrow(x_position, y_position, target_x_position - x_position - 0.001,
                                    target_y_position - y_position - 0.04,
                                    head_width=0.04, head_length=0.04, fc='k', ec='k', length_includes_head=True)
                        break
                    if starts_levels[j] <= starts_levels[i]:
                        break


            x_position = ends[-1]
            y_position = ends_levels[-1]
            yvalues[-1] = y_position
            for j in reversed(range(len(starts))):
                if ends_levels[-1] + 1 == starts_levels[j]:
                    target_x_position = starts[j]
                    target_y_position = starts_levels[j]
                    axis.arrow(x_position, y_position, target_x_position - x_position + 0.001,
                                target_y_position - y_position - 0.04,
                                head_width=0.04, head_length=0.04, fc='k', ec='k', length_includes_head=True)
                    break
                if ends_levels[-1] >= starts_levels[j]:
                    break



            xValues = starts + ends[-1:]
            axis.plot(xValues, yvalues, 'bo', markersize=markersize, color="black")
            axis.set_xlim([self.a[d + offset]-0.01, self.b[d + offset]+0.01])
            axis.set_ylim([-0.04, max_level+0.04])
            axis.set_ylim(axis.get_ylim()[::-1])
            axis.set_yticks(list(list(range(max_level+1))))
            axis.set_ylabel("level")
            axis.set_title("$x_" + str(d + 1 + offset) + "$")
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
        plt.show()
        return fig

    def init_evaluation_operation(self, areas):
        self.operation.initialize_refinement_container_dimension_wise(areas[0])

    def evaluate_operation_area(self, component_grid:ComponentGridInfo, area, additional_info=None):
        if self.grid.is_global():
            # get 1d coordinates of the grid points that define the grid; they are calculated based on the levelvector
            gridPointCoordsAsStripes, grid_point_levels, children_indices = self.get_point_coord_for_each_dim(component_grid.levelvector)

            # calculate the operation on the grid
            integral = self.operation.calculate_operation_dimension_wise(gridPointCoordsAsStripes, grid_point_levels, component_grid, self.a, self.b, False)#self.refinements != 0 and not self.do_high_order and not self.grid.modified_basis)

            # compute the error estimates for further refining the Refinementobjects and therefore the future grid
            self.operation.compute_error_estimates_dimension_wise(gridPointCoordsAsStripes, grid_point_levels, children_indices, component_grid)

            # save the number of evaluations used per d-1 dimensional slice
            #for d in range(self.dim):
            #    factor = component_grid.coefficient if self.grid.isNested() else 1
            #    self.evaluationCounts[d][component_grid.levelvector[d] - 1] += factor * np.prod([self.grid.numPoints[d2] if d2 != d else 1 for d2 in range(self.dim)])
            return np.prod(self.grid.numPoints)
        else:
            pass

    # This method computes additional values after the compution of the integrals for the current
    # refinement step is finished. This method is executed before the refinement process.
    def finalize_evaluation_operation(self, areas, evaluation_array):
        super().finalize_evaluation_operation(areas, evaluation_array)
        #self.refinement.print_containers_only()
        #if self.version == 1:
        #    for d in range(self.dim):
        #        container_d = self.refinement.get_refinement_container_for_dim(d)
        #        for area in container_d.get_objects():
        #            level = max(area.levels)
        #            area.set_evaluations(np.sum(self.evaluationCounts[d][level-1:]))

    def _initialize_points(self, points, func_mid, d, i1, i2):
        if i1+1 >= i2:
            return
        i = (i1 + i2) // 2
        points[i] = func_mid(points[i1], points[i2], d)
        self._initialize_points(points, func_mid, d, i1, i)
        self._initialize_points(points, func_mid, d, i, i2)

    def _initialize_levels(self, levels, i1, i2, level):
        if i1+1 >= i2:
            return
        i = (i1 + i2) // 2
        level = level+1
        levels[i] = level
        self._initialize_levels(levels, i1, i, level)
        self._initialize_levels(levels, i, i2, level)

    def initialize_refinement(self):
        initial_points = []
        maxv = self.lmax[0]
        assert maxv > 1
        assert all([l == maxv for l in self.lmax])
        num_points = 2 ** maxv + 1
        levels = [0 for _ in range(num_points)]
        self._initialize_levels(levels, 0, num_points-1, 0)
        func_mid = self.grid.get_mid_point
        for d in range(self.dim):
            points = [None for _ in range(num_points)]
            points[0] = self.a[d]
            points[num_points-1] = self.b[d]
            if self.chebyshev_points:
                points = np.linspace(0,1, 2**maxv + 1)
                points = [self.a[d] + (self.b[d]- self.a[d]) * (1 - math.cos(p * math.pi)) / 2 for p in points]
            else:
                self._initialize_points(points, func_mid, d, 0, num_points-1)
            initial_points.append(np.array(points))
        self.refinement = MetaRefinementContainer([RefinementContainer
                                                   ([RefinementObjectSingleDimension(initial_points[d][i],
                                                                                     initial_points[d][i + 1], d, self.dim, list((levels[i], levels[i+1])), grid=self.grid,
                                                                                     coarsening_level=0, a=self.a[d], b=self.b[d], chebyshev=self.chebyshev_points) for i in
                                                     range(2 ** maxv)], d, self.errorEstimator) for d in
                                                   range(self.dim)])
        if self.dim_adaptive:
            self.combischeme.init_adaptive_combi_scheme(self.lmax[0], self.lmin[0])
        #self.evaluationCounts = [np.zeros(self.lmax[d]) for d in range(self.dim)]
        if self.operation is not None:
            self.operation.init_dimension_wise(self.grid, self.grid_surplusses, self.refinement, self.lmin, self.lmax, self.a, self.b, self.version)


    def get_areas(self):
        if (self.grid.is_global() == True):
            return [self.refinement]
        assert False
        # get a list of lists which contains range(refinements[d]) for each dimension d where the refinements[d] are the number of subintervals in this dimension
        #indices = [list(range(len(refineDim))) for refineDim in self.refinement.get_new_objects()]
        # this command creates tuples of size this_dim of all combinations of indices (e.g. this_dim = 2 indices = ([0,1],[0,1,2,3]) -> areas = [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3)] )
        #return list(zip(*[g.ravel() for g in np.meshgrid(*indices)]))

    def get_new_areas(self):
        return self.get_areas()

    def do_refinement(self, area, position):
        # print("-------------------\nREFINING", position)
        lmaxChange = self.refinement.refine(position)
        # the following is currently solved by initializing all data structures anew before each evalute_integral()
        refinement_dim = position[0]
        #max_level = self.rebalance(refinement_dim)
        #if lmaxChange is not None and max_level <= self.lmax[refinement_dim]:
        #    lmaxChange = None
        #if lmaxChange is not None:
        #    for d in range(self.dim):
        #        if lmaxChange[d] != 0:
        #            self.raise_lmax(d, lmaxChange[d])
        #    #print("New scheme:")
        #    self.scheme = self.combischeme.getCombiScheme(self.lmin[0], self.lmax[0], do_print=False)
        #    return False
        return False

    def raise_lmax(self, d: int, value: int):
        self.lmax[d] += value
        if self.dim_adaptive:
            if self.print_output:
                print("New lmax:", self.lmax)
            while (True):
                refinements = 0
                active_indices = set(self.combischeme.get_active_indices())
                for index in active_indices:
                    if max(self.lmax) + self.dim - 1 > sum(index) and all(
                            [self.lmax[d] > index[d] for d in range(self.dim)]):
                        self.combischeme.update_adaptive_combi(index)
                        refinements += 1
                if refinements == 0:
                    break

    def rebalance(self, d):
        refinement_container = self.refinement.get_refinement_container_for_dim(d)
        self.rebalance_interval(0, refinement_container.size(), 1, refinement_container)
        #refinement_container.printContainer()

    def rebalance_interval(self, start, end, level, refinement_container):
        if end - start <= 2:
            return
        refineContainer = refinement_container
        position_level = None
        position_level_1_left = None
        position_level_1_right = None
        for i, refinement_object in enumerate(refineContainer.get_objects()[start:end]):
            if refinement_object.levels[1] == level:
                position_level = i
            if refinement_object.levels[1] == level + 1:
                if position_level_1_left is None:
                    position_level_1_left = i
                else:
                    assert position_level_1_right is None
                    position_level_1_right = i
                    break

        #refineContainer.printContainer()
        #print(refinement_object.this_dim, position_level, position_level_1_left, position_level_1_right, start, end, level )
        safetyfactor = 10**-1#0#0.1
        assert position_level is not None
        assert position_level_1_right is not None
        assert position_level_1_left is not None
        new_leaf_reached = False
        #print(i+2, end - start + 1, (i + 2) / (end - start + 1), i, start, end, level)
        if position_level_1_right is not None and abs((position_level) / (end-start - 2) - 0.5) > abs((position_level_1_right) / (end-start - 2) - 0.5) + safetyfactor:
            position_new_leaf = None
            if self.print_output:
                print("Rebalancing!")
            for j, refinement_object in enumerate(refineContainer.get_objects()[start:end]):
                if j < end - start - 1:
                    next_refinement_object = refineContainer.get_object(j+1+start)
                    if j <= position_level:
                        refinement_object.levels[1] += 1
                        next_refinement_object.levels[0] += 1
                    elif j == position_level:
                        refinement_object.levels[1] += 1
                        next_refinement_object.levels[0] += 1

                    else:
                        if refinement_object.levels[1] == level + 1:
                            new_leaf_reached = True
                            position_new_leaf = start + j
                        if j > position_level and new_leaf_reached:
                            refinement_object.levels[1] -= 1
                            next_refinement_object.levels[0] -= 1
            assert position_new_leaf is not None
            self.rebalance_interval(start,position_new_leaf + 1, level + 1, refinement_container)
            self.rebalance_interval(position_new_leaf + 1, end, level + 1, refinement_container)
            return
            #refineContainer.printContainer()

        new_leaf_reached = True
        if position_level_1_left is not None and abs((position_level) / (end-start - 2) - 0.5) > abs((position_level_1_left) / (end-start - 2) - 0.5) + safetyfactor:
            position_new_leaf = None
            if self.print_output:
                print("Rebalancing!")
            for j, refinement_object in enumerate(refineContainer.get_objects()[start:end]):
                if j < end - start - 1:
                    next_refinement_object = refineContainer.get_object(j+1+start)

                    if j >= position_level:
                        refinement_object.levels[1] += 1
                        next_refinement_object.levels[0] += 1

                    elif j == position_level:
                        refinement_object.levels[1] += 1
                        next_refinement_object.levels[0] += 1

                    else:
                        if j < position_level and new_leaf_reached:
                            refinement_object.levels[1] -= 1
                            next_refinement_object.levels[0] -= 1
                        if refinement_object.levels[1] == level:
                            new_leaf_reached = False
                            position_new_leaf = start + j
            assert position_new_leaf is not None
            self.rebalance_interval(start, position_new_leaf + 1, level + 1, refinement_container)
            self.rebalance_interval(position_new_leaf + 1, end, level + 1, refinement_container)
            return

        self.rebalance_interval(start, start + position_level + 1, level + 1, refinement_container)
        self.rebalance_interval(start + position_level + 1, end, level + 1, refinement_container)

            #refineContainer.printContainer()

    def update_coarsening_values(self, refinement_container_d, d):
        update_dimension = 0
        for refinement_object in refinement_container_d.get_objects():
            refinement_object.coarsening_level = self.lmax[d] - max(refinement_object.levels)
            if refinement_object.coarsening_level < update_dimension :
                update_dimension = refinement_object.coarsening_level
            #assert refinement_object.coarsening_level >= -1
        return update_dimension * -1

    def refinement_postprocessing(self):
        self.subtraction_value_cache = {}
        self.max_level_dict = {}
        self.refinement.apply_remove(sort=True)
        self.refinement.refinement_postprocessing()
        self.refinement.reinit_new_objects()
        #self.evaluationCounts = [np.zeros(self.lmax[d]) for d in range(self.dim)]
        if self.rebalancing:
            for d in range(self.dim):
                self.rebalance(d)
        for d in range(self.dim):
            refinement_container_d = self.refinement.get_refinement_container_for_dim(d)
            update_d = self.update_coarsening_values(refinement_container_d, d)
            if update_d > 0:
                self.raise_lmax(d, update_d)
                refinement_container_d.update_values(update_d)
        self.scheme = self.combischeme.getCombiScheme(do_print=False)


class NodeInfo(object):
    def __init__(self, child, left_parent, right_parent, left_parent_of_left_parent, right_parent_of_right_parent, has_left_child, has_right_child, left_refinement_object, right_refinement_object, level_child):
        self.child = child
        self.left_parent = left_parent
        self.right_parent = right_parent
        self.left_parent_of_left_parent = left_parent_of_left_parent
        self.right_parent_of_right_parent = right_parent_of_right_parent
        self.has_left_child = has_left_child
        self.has_right_child = has_right_child
        self.left_refinement_object = left_refinement_object
        self.right_refinement_object = right_refinement_object
        self.level_child = level_child

