from spatiallyAdaptiveBase import *
from Grid import *
import numpy.testing as npt
def sortToRefinePosition(elem):
    # sort by depth
    return elem[1]


class SpatiallyAdaptiveSingleDimensions2(SpatiallyAdaptivBase):
    def __init__(self, a, b, norm=np.inf, dim_adaptive=False, version=0):
        self.grid = GlobalTrapezoidalGrid(a, b, boundary=True)
        SpatiallyAdaptivBase.__init__(self, a, b, self.grid, norm=norm)
        self.dim_adaptive = dim_adaptive
        self.evaluationCounts = None
        self.version = version
        self.dict_integral = {}
        self.dict_points = {}
        self.no_previous_integrals = True

    def coarsen_grid(self, area, levelvec):
        pass

    # returns the points coordinates of a single component grid with refinement
    def get_points_all_dim(self, levelvec, numSubDiagonal):
        indicesList, children_indices = self.get_point_coord_for_each_dim(levelvec)
        # this command creates tuples of size this_dim of all combinations of indices (e.g. this_dim = 2 indices = ([0,1],[0,1,2,3]) -> areas = [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3)] )
        allPoints = list(set(zip(*[g.ravel() for g in np.meshgrid(*indicesList)])))
        return allPoints

    # returns the points of a single component grid with refinement
    def get_points_component_grid(self, levelvec, numSubDiagonal):
        return self.get_points_all_dim(levelvec, numSubDiagonal)

    # returns list of coordinates for each dimension (basically refinement stripes)
    def get_point_coord_for_each_dim(self, levelvec):
        refinement = self.refinement
        # get a list of all coordinates for every this_dim (so (0, 1), (0, 0.5, 1) for example)
        indicesList = []
        children_indices = []
        for d in range(0, self.dim):
            refineContainer = refinement.get_refinement_container_for_dim(d)
            indicesDim = []
            children_indices_dim = []
            indicesDim.append(refineContainer.get_objects()[0].start)
            for i in range(len(refineContainer.get_objects())):
                refineObj = refineContainer.get_objects()[i]
                if i + 1 < len(refineContainer.get_objects()):
                    next_refineObj = refineContainer.get_objects()[i + 1]
                else:
                    next_refineObj = None
                if self.version == 2:
                    refineObj_temp = refineObj
                    max_level = refineObj_temp.levels[1]
                    k = 1
                    while(i - k > 0):
                        refineObj_temp = refineContainer.get_objects()[i - k]
                        max_level = max(max_level, refineObj_temp.levels[0])
                        if refineObj_temp.levels[0] <= refineObj.levels[1]:
                            break
                        k += 1
                    k = 1
                    while(i + k < len(refineContainer.get_objects())):
                        max_level = max(max_level, refineObj_temp.levels[1])
                        refineObj_temp = refineContainer.get_objects()[i + k]
                        if refineObj_temp.levels[1] <= refineObj.levels[1]:
                            break
                        k += 1
                    subtraction_value = self.lmax[d] - max_level
                else:
                    subtraction_value = 0
                if (refineObj.levels[1] <= max(levelvec[d] - subtraction_value, 1)):
                    indicesDim.append(refineObj.end)
                    if next_refineObj is not None and self.is_child(refineObj, next_refineObj):
                        children_indices_dim.append(self.get_node_info(refineObj, next_refineObj))
            indicesList.append(indicesDim)
            children_indices.append(children_indices_dim)
        return indicesList, children_indices

    def is_child(self, refineObj, next_refineObj):
        if refineObj.levels[0] < refineObj.levels[1] or next_refineObj.levels[1] < refineObj.levels[1]:
            return True
        else:
            return False

    def get_node_info(self, refineObj, next_refineObj):
        child = refineObj.end
        right_refinement_object = None
        left_refinement_object = None
        if refineObj.levels[0] < refineObj.levels[1]:
            left_parent = refineObj.start
            left_child = False
            left_refinement_object = refineObj
            if next_refineObj.levels[1] < refineObj.levels[1]:
                right_parent = next_refineObj.end
                right_child = False
                right_refinement_object = next_refineObj
            else:
                right_parent = child + (child - left_parent)
                right_child = True
        else:
            left_child = True
            assert next_refineObj.levels[1] < refineObj.levels[1]
            right_child = False
            right_refinement_object = next_refineObj
            right_parent = next_refineObj.end
            left_parent = child - (right_parent - child)
        npt.assert_almost_equal(right_parent - child, child - left_parent, decimal=10)
        return NodeInfo(child, left_parent, right_parent, left_child, right_child, left_refinement_object, right_refinement_object)

    # this method draws the 1D refinement of each dimension individually
    def draw_refinement(self, filename=None):  # update with meta container
        plt.rcParams.update({'font.size': 32})
        refinement = self.refinement
        dim = self.dim
        fig, ax = plt.subplots(ncols=1, nrows=dim, figsize=(20, 10))
        for d in range(dim):
            starts = [refinementObject.start for refinementObject in refinement.refinementContainers[d].get_objects()]
            ends = [refinementObject.end for refinementObject in refinement.refinementContainers[d].get_objects()]
            for i in range(len(starts)):
                ax[d].add_patch(
                    patches.Rectangle(
                        (starts[i], -0.1),
                        ends[i] - starts[i],
                        0.2,
                        fill=False  # remove background
                    )
                )
            xValues = starts + ends
            yValues = np.zeros(len(xValues))
            ax[d].plot(xValues, yValues, 'bo', markersize=10, color="black")
            ax[d].set_xlim([self.a[d], self.b[d]])
            ax[d].set_ylim([-0.1, 0.1])
            ax[d].set_yticks([])
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
        plt.show()
        return fig

    # evaluate the integral of f in a specific area with numPoints many points using the specified integrator set in the grid
    # We also interpolate the function to the finest width to calculate the error of the combination in each
    def evaluate_area(self, f, area, component_grid):
        if self.grid.is_global():
            gridPointCoordsAsStripes, children_indices = self.get_point_coord_for_each_dim(component_grid.levelvector)
            start = self.a
            end = self.b
            if self.no_previous_integrals:
                self.grid.set_grid(gridPointCoordsAsStripes)
                integral = self.grid.integrator(f, self.grid.numPoints, start, end)
                if sum(component_grid.levelvector) == max(self.lmax) + self.dim - 1 or tuple(component_grid.levelvector) in self.combischeme.get_active_indices():
                    self.calculate_surplusses(gridPointCoordsAsStripes, children_indices)
                for d in range(self.dim):
                    factor = component_grid.coefficient if self.grid.isNested() else 1
                    self.evaluationCounts[d][component_grid.levelvector[d] - 1] += factor * np.prod([self.grid.numPoints[d2] if d2 != d else 1 for d2 in range(self.dim)])
            else:
                #print("Calculating for", component_grid.levelvector)
                previous_integral, previous_points = self.get_previous_integral_and_points(component_grid.levelvector)
                #print(previous_points, gridPointCoordsAsStripes)
                integral = np.array(previous_integral)
                previous_points_coarsened = list(previous_points)
                #print("Previous integral", previous_integral)
                #self.grid.set_grid(previous_points)
                #integral2 = self.grid.integrator(f, self.grid.numPoints, start, end)
                #npt.assert_almost_equal(integral, integral2, decimal=12)
                modification_points, modification_points_coarsen = self.get_modification_points(previous_points, gridPointCoordsAsStripes)
                if modification_points_coarsen is not None:
                    for d in range(self.dim):
                        previous_points_coarsened[d] = list(previous_points[d])
                        for mod_point in modification_points_coarsen[d]:
                            #print(mod_point, previous_points_coarsened[d])
                            for removal_point in mod_point[1]:
                                previous_points_coarsened[d].remove(removal_point)
                    #print(integral, self.subtract_contributions(modification_points_coarsen, previous_points_coarsened, previous_points),
                    #      self.get_new_contributions(modification_points_coarsen, previous_points))
                    #self.grid.set_grid(previous_points_coarsened)
                    #print(list(zip(*self.grid.get_points_and_weights())))
                    #self.grid.set_grid(previous_points)
                    #print(list(zip(*self.grid.get_points_and_weights())))
                    integral += self.subtract_contributions(modification_points_coarsen, previous_points_coarsened, previous_points)
                    integral -= self.get_new_contributions(modification_points_coarsen, previous_points)
                    #print(previous_points, previous_points_coarsened)
                #self.grid.set_grid(previous_points_coarsened)
                #integral2 = self.grid.integrator(f, self.grid.numPoints, start, end)
                #npt.assert_almost_equal(integral, integral2, decimal=12)
                if modification_points is not None:
                    integral -= self.subtract_contributions(modification_points, previous_points_coarsened, gridPointCoordsAsStripes)
                    integral += self.get_new_contributions(modification_points, gridPointCoordsAsStripes)
                    #self.update_surplusses()
                #self.grid.set_grid(gridPointCoordsAsStripes)
                #integral2 = self.grid.integrator(f, self.grid.numPoints, start, end)
                #npt.assert_almost_equal(integral, integral2, decimal=12)
                if sum(component_grid.levelvector) == max(self.lmax) + self.dim - 1 or tuple(
                        component_grid.levelvector) in self.combischeme.get_active_indices():
                    self.grid.set_grid(gridPointCoordsAsStripes)
                    self.calculate_surplusses(gridPointCoordsAsStripes, children_indices)

            self.dict_integral[tuple(component_grid.levelvector)] = integral
            self.dict_points[tuple(component_grid.levelvector)] = gridPointCoordsAsStripes
            #print("Children indices", children_indices)
            #print("integral", component_grid.levelvector, integral )#, self.dict_integral.items())
            return integral, None, np.prod(self.grid.numPoints)
        else:
            pass

    def get_previous_integral_and_points(self, levelvector):
        if tuple(levelvector) in self.dict_integral:
            #print("Found for", levelvector, "old entry for", levelvector)
            return self.dict_integral[tuple(levelvector)], self.dict_points[tuple(levelvector)]
        else:
            k = 1
            dimensions = []
            for d in range(self.dim):
                if self.lmax[d] - k > 0:
                    dimensions.append(d)
            while k < max(self.lmax):
                dimensions_new = []
                for d in dimensions:
                    if self.lmax[d] - k >= 0:
                        dimensions_new.append(d)
                for d in dimensions_new:
                    levelvec_temp = list(levelvector)
                    levelvec_temp[d] -= k
                    if tuple(levelvec_temp) in self.dict_integral:
                        #print("Found for", levelvector, "old entry for", levelvec_temp)
                        return self.dict_integral[tuple(levelvec_temp)], self.dict_points[tuple(levelvec_temp)]
                k += 1
        #print(self.dict_integral.keys(), levelvector)
        assert False

    def get_modification_points(self, old_points, new_points):
        found_modification = found_modification2 = False
        modification_array = [[] for d in range(self.dim)]
        modification_array2 = [[] for d in range(self.dim)]

        for d in range(self.dim):
            modifications = sorted(list(set(new_points[d]) - set(old_points[d])))
            if len(modifications) != 0:
                #print(modifications, new_points[d], old_points[d])
                found_modification = True
                modification_1D = self.get_modification_objects(modifications, new_points[d])
                modification_array[d].extend(list(modification_1D))

            modifications_coarsen = sorted(list(set(old_points[d]) - set(new_points[d])))
            if len(modifications_coarsen) != 0:
                found_modification2 = True
                modification_1D = self.get_modification_objects(modifications_coarsen, old_points[d])
                modification_array2[d].extend(list(modification_1D))
        #if found_modification or found_modification2:
            #print("Found modification!", modification_array if found_modification else None,  modification_array2 if found_modification2 else None)
        return modification_array if found_modification else None, modification_array2 if found_modification2 else None

    def get_modification_objects(self, modifications, grid_points):
        modification_1D = []
        k = 0
        for i in range(len(grid_points)):
            if grid_points[i] == modifications[k]:
                j = 1
                while (k + j < len(modifications) and grid_points[i + j] == modifications[k + j]):
                    j += 1
                modification_1D.append((grid_points[i - 1], list(grid_points[i:i + j]), grid_points[i + j]))
                k += j
                if k == len(modifications):
                    break
        return modification_1D

    def subtract_contributions(self, modification_points, old_points, new_points):
        self.grid.set_grid(new_points)

        weights = self.grid.weights
        dict_weights_fine = [{} for d in range(self.dim)]
        for d in range(self.dim):
            for p, w in zip(new_points[d], weights[d]):
                #print(p,w)
                dict_weights_fine[d][p] = w
        self.grid.set_grid(old_points)

        integral = 0.0
        for d in range(self.dim):
            for point in modification_points[d]:
                integral += self.calc_slice_through_points([point[0],point[2]], old_points, d, modification_points, use_factor=True, dict=dict_weights_fine)
        return integral

    def get_new_contributions(self, modification_points, new_points):
        self.grid.set_grid(new_points)
        integral = 0.0
        for d in range(self.dim):
            for point in modification_points[d]:
                integral += self.calc_slice_through_points(point[1], new_points, d, modification_points)
        return integral

    def calc_slice_through_points(self, points_for_slice, grid_points, d, modification_points, use_factor=False, dict=None):
        integral = 0.0
        #print(grid_points[d], points_for_slice)
        positions = [grid_points[d].index(point) for point in points_for_slice]
        points = list(zip(*[g.ravel() for g in np.meshgrid(*[grid_points[d2] if d != d2 else points_for_slice for d2 in range(self.dim)])]))
        indices = list(zip(*[g.ravel() for g in np.meshgrid(*[range(len(grid_points[d2])) if d != d2 else positions for d2 in range(self.dim)])]))
        for i in range(len(points)):
            index = indices[i]
            current_point = points[i]
            #print("Point", current_point)
            weight = self.grid.getWeight(index)
            if use_factor:
                '''
                value2 = (points_for_slice[1] - points_for_slice[0]) * 0.25
                value3 = (points_for_slice[1] - points_for_slice[0]) * 0.5
                #print(self.grid.weights[d][index[d]], value2)
                #assert value3 == 0.5 * self.grid.weights[d][index[d]]
                for mod_point in modification_points[d]:
                    if points_for_slice[0] == mod_point[0] and points_for_slice[1] == mod_point[2]:
                        value = self.get_value(current_point[d], mod_point)
                #print(current_point[d], mod_point, points_for_slice)
                #assert(value == value2)
                number_of_dimensions_that_intersect = 1
                for d2 in range(self.dim):
                    if d2 == d:
                        continue
                    modification_found = False
                    for mod_point in modification_points[d2]:
                        if current_point[d2] in mod_point:
                            assert current_point[d2] not in mod_point[1]
                            #print(current_point, mod_point)
                            value2 *= (mod_point[2] - mod_point[0]) * 0.25
                            value *= (self.grid.weights[d2][index[d2]] + self.get_value(current_point[d2], mod_point)) - (mod_point[2] - mod_point[0]) * 0.5
                            value3 *= (mod_point[2] - mod_point[0]) * 0.5
                            number_of_dimensions_that_intersect += 1
                            modification_found = True
                            break
                    if not modification_found:
                        value *= self.grid.weights[d2][index[d2]]
                        value2 *= self.grid.weights[d2][index[d2]]
                        value3 *= self.grid.weights[d2][index[d2]]
                '''
                value = 1
                for d in range(self.dim):
                    value *= dict[d][current_point[d]]
                #print("Intersections", number_of_dimensions_that_intersect, current_point)
                number_of_dimensions_that_intersect = 0
                for d2 in range(self.dim):
                    for mod_point in modification_points[d2]:
                        if current_point[d2] == mod_point[0] or current_point[d2] == mod_point[2]:
                            number_of_dimensions_that_intersect += 1
                factor = (weight - value)/number_of_dimensions_that_intersect
                #print(current_point, value, weight, factor)
                #if number_of_dimensions_that_intersect >= 1:
                    #assert(value == value2)
                    #print("Factor", current_point, factor, weight, value, weight/value, value2, weight/value2, value3, value3/value, number_of_dimensions_that_intersect)
                #assert(value3 == 2**number_of_dimensions_that_intersect * value)
            else:
                number_of_dimensions_that_intersect = 1
                for d2 in range(self.dim):
                    if d2 == d:
                        continue
                    for mod_point in modification_points[d2]:
                        if current_point[d2] in mod_point[1]:
                            number_of_dimensions_that_intersect += 1
                factor = weight / number_of_dimensions_that_intersect
                #if number_of_dimensions_that_intersect > 1:
                    #print("Factor", current_point, factor, weight, number_of_dimensions_that_intersect)

            assert(factor > 0)
            integral += self.f(current_point) * factor
        #print(integral)
        return integral

    def get_value(self, point_coord_d, mod_point):
        #print(mod_point)
        value = None
        if point_coord_d == mod_point[0]:
            value = (mod_point[1][0] - mod_point[0]) * 0.5
        if point_coord_d == mod_point[2]:
            value = (mod_point[2] - mod_point[1][-1]) * 0.5
        #print(value)
        #assert value == (mod_point[2] - mod_point[0]) * 0.25
        #print(point_coord_d, mod_point)
        assert value is not None
        return value

    def finalize_evaluation(self):
        self.no_previous_integrals = False
        if self.version == 1:
            for d in range(self.dim):
                container_d = self.refinement.get_refinement_container_for_dim(d)
                for area in container_d.get_objects():
                    level = max(area.levels)
                    #print("Evaluations for dim", d, "is", np.sum(self.evaluationCounts[d][level-1:]), "for area", area.start, area.end)
                    area.set_evaluations(np.sum(self.evaluationCounts[d][level-1:]))

    def calculate_surplusses(self, grid_points, children_indices):
        for d in range(0, self.dim):
            refineContainer = self.refinement.get_refinement_container_for_dim(d)
            refinement_objects = refineContainer.get_objects()
            for child_info in children_indices[d]:
                left_parent = child_info.left_parent
                right_parent = child_info.right_parent
                child = child_info.child
                volume, evaluations = self.sum_up_volumes_for_point(left_parent=left_parent, right_parent=right_parent, child=child, grid_points=grid_points, d=d)
                if not child_info.has_right_child:
                    child_info.right_refinement_object.add_volume(volume / 2.0)
                    child_info.right_refinement_object.add_evaluations(evaluations / 2.0)
                if not child_info.has_left_child:
                    child_info.left_refinement_object.add_volume(volume/2.0)
                    child_info.left_refinement_object.add_evaluations(evaluations / 2.0)

    def sum_up_volumes_for_point(self, left_parent, right_parent, child, grid_points, d):
        volume = 0.0
        assert right_parent > child > left_parent
        npt.assert_almost_equal(right_parent - child, child - left_parent, decimal=10)
        points_left_parent = list(zip(*[g.ravel() for g in np.meshgrid(*[grid_points[d2] if d != d2 else [left_parent] for d2 in range(self.dim)])]))
        points_right_parent = list(zip(*[g.ravel() for g in np.meshgrid(*[grid_points[d2] if d != d2 else [right_parent] for d2 in range(self.dim)])]))
        points_children = list(zip(*[g.ravel() for g in np.meshgrid(*[grid_points[d2] if d != d2 else [child] for d2 in range(self.dim)])]))
        indices = list(zip(*[g.ravel() for g in np.meshgrid(*[range(len(grid_points[d2])) if d != d2 else None for d2 in range(self.dim)])]))
        for i in range(len(points_children)):
            index = indices[i]
            factor = np.prod([self.grid.weights[d2][index[d2]] if d2 != d else 1 for d2 in range(self.dim)])
            volume += factor * abs(self.f(points_children[i]) - 0.5 * (self.f(points_left_parent[i]) + self.f(points_right_parent[i]))) * (right_parent - child)
        if self.version == 0:
            evaluations = len(points_right_parent)
        else:
            evaluations = 0
        return abs(volume), evaluations

    def initialize_refinement(self):
        initial_points = []
        for d in range(self.dim):
            initial_points.append(np.linspace(self.a[d], self.b[d], 2 ** 1 + 1))
        self.refinement = MetaRefinementContainer([RefinementContainer
                                                   ([RefinementObjectSingleDimension(initial_points[d][i],
                                                                                     initial_points[d][i + 1], d, self.dim, (i % 2, (i+1) % 2),
                                                                                     self.lmax[d] - 1, dim_adaptive=self.dim_adaptive) for i in
                                                     range(2 ** 1)], d, self.errorEstimator) for d in
                                                   range(self.dim)])
        if self.dim_adaptive:
            self.combischeme.init_adaptive_combi_scheme(self.lmax[0], self.lmin[0])
        self.evaluationCounts = [np.zeros(self.lmax[d]) for d in range(self.dim)]


    def get_areas(self):
        if (self.grid.is_global() == True):
            return [self.refinement]
        # get a list of lists which contains range(refinements[d]) for each dimension d where the refinements[d] are the number of subintervals in this dimension
        indices = [list(range(len(refineDim))) for refineDim in self.refinement.get_new_objects()]
        # this command creates tuples of size this_dim of all combinations of indices (e.g. this_dim = 2 indices = ([0,1],[0,1,2,3]) -> areas = [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3)] )
        return list(zip(*[g.ravel() for g in np.meshgrid(*indices)]))

    def get_new_areas(self):
        return self.get_areas()

    def do_refinement(self, area, position):
        # print("-------------------\nREFINING", position)
        lmaxChange = self.refinement.refine(position)
        # the following is currently solved by initializing all data structures anew before each evalute_integral()
        if lmaxChange is not None:
            self.lmax = [self.lmax[d] + lmaxChange[d] for d in range(self.dim)]
            if self.dim_adaptive:
                print("New lmax:", self.lmax)
                while(True):
                    refinements = 0
                    active_indices = set(self.combischeme.get_active_indices())
                    for index in active_indices:
                        if max(self.lmax) + self.dim - 1  > sum(index) and all([self.lmax[d] > index[d] for d in range(self.dim)]):
                            self.combischeme.update_adaptive_combi(index)
                            refinements +=1
                    if refinements == 0:
                        break
            print("New scheme:")
            self.scheme = self.combischeme.getCombiScheme(self.lmin[0], self.lmax[0], do_print=False)
            return False
        return False

    def refinement_postprocessing(self):
        self.refinement.apply_remove(sort=True)
        self.refinement.refinement_postprocessing()
        self.refinement.reinit_new_objects()
        self.evaluationCounts = [np.zeros(self.lmax[d]) for d in range(self.dim)]


class NodeInfo(object):
    def __init__(self, child, left_parent, right_parent, has_left_child, has_right_child, left_refinement_object, right_refinement_object):
        self.child = child
        self.left_parent = left_parent
        self.right_parent = right_parent
        self.has_left_child = has_left_child
        self.has_right_child = has_right_child
        self.left_refinement_object = left_refinement_object
        self.right_refinement_object = right_refinement_object

