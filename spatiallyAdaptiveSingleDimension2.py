from spatiallyAdaptiveBase import *
from Grid import *
import numpy.testing as npt
from GridOperation import *

def sortToRefinePosition(elem):
    # sort by depth
    return elem[1]


class SpatiallyAdaptiveSingleDimensions2(SpatiallyAdaptivBase):
    def __init__(self, a, b, norm=np.inf, dim_adaptive=True, version=2, do_high_order=False, max_degree=1000, split_up=True, do_nnls=False, boundary = True, modified_basis=False, operation=None, margin=None):
        self.do_high_order = do_high_order
        if self.do_high_order:
            self.grid = GlobalHighOrderGrid(a, b, boundary=boundary, max_degree=max_degree, split_up=split_up, do_nnls=do_nnls)
            self.grid_surplusses = GlobalTrapezoidalGrid(a, b, boundary=boundary, modified_basis=modified_basis) # GlobalHighOrderGrid(a, b, boundary=True, max_degree=max_degree, split_up=split_up, do_nnls=do_nnls) #GlobalTrapezoidalGrid(a, b, boundary=True)

        else:
            # do_high_order is not yet supported for UQ
            if isinstance(operation, UncertaintyQuantification):
                self.grid = GlobalTrapezoidalGridWeighted(a, b, operation, boundary=boundary)
                self.grid_surplusses = GlobalTrapezoidalGridWeighted(a, b, operation, boundary=boundary)
            else:
                self.grid = GlobalTrapezoidalGrid(a, b, boundary=boundary, modified_basis=modified_basis)
                self.grid_surplusses = GlobalTrapezoidalGrid(a, b, boundary=boundary, modified_basis=modified_basis)

        SpatiallyAdaptivBase.__init__(self, a, b, self.grid, norm=norm)
        self.dim_adaptive = dim_adaptive
        self.evaluationCounts = None
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
        self.equidistant = True

    def interpolate_points(self, interpolation_points, component_grid):
        gridPointCoordsAsStripes, children_indices = self.get_point_coord_for_each_dim(component_grid.levelvector)
        return Interpolation.interpolate_points(self.f, self.dim, self.grid, gridPointCoordsAsStripes, interpolation_points)

    def coarsen_grid(self, area, levelvec):
        pass

    # returns the points coordinates of a single component grid with refinement
    def get_points_all_dim(self, levelvec, numSubDiagonal):
        indicesList, children_indices = self.get_point_coord_for_each_dim(levelvec)
        if not self.grid.boundary:
            indicesList = [indices[1:-1] for indices in indicesList]
        # this command creates tuples of size this_dim of all combinations of indices (e.g. this_dim = 2 indices = ([0,1],[0,1,2,3]) -> areas = [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3)] )
        allPoints = list(set(zip(*[g.ravel() for g in np.meshgrid(*indicesList)])))
        return allPoints

    # returns the points of a single component grid with refinement
    def get_points_component_grid(self, levelvec, numSubDiagonal):
        return self.get_points_all_dim(levelvec, numSubDiagonal)

    def get_points_and_weights_component_grid(self, levelvec, numSubDiagonal):
        point_coords, _ =self.get_point_coord_for_each_dim(levelvec)
        self.grid.set_grid(point_coords)
        points, weights = self.grid.get_points_and_weights()
        return points, weights

    # returns list of coordinates for each dimension (basically refinement stripes) + all points that are associated
    # with a child in the global refinement structure. There might be now such points that correspond to a global child.
    def get_point_coord_for_each_dim(self, levelvec):
        refinement = self.refinement
        # get a list of all coordinates for every this_dim (so (0, 1), (0, 0.5, 1) for example)
        indicesList = []
        children_indices = []
        for d in range(0, self.dim):
            refineContainer = refinement.get_refinement_container_for_dim(d)
            indicesDim = []
            indices_levelDim = []

            children_indices_dim = []
            indicesDim.append(refineContainer.get_objects()[0].start)
            indices_levelDim.append(refineContainer.get_objects()[0].levels[0])
            for i in range(len(refineContainer.get_objects())):
                refineObj = refineContainer.get_objects()[i]
                if i + 1 < len(refineContainer.get_objects()):
                    next_refineObj = refineContainer.get_objects()[i + 1]
                else:
                    next_refineObj = None
                if self.version == 2:
                    refineObj_temp = refineObj
                    max_level = refineObj_temp.levels[1]
                    k = 0
                    while(i - k > 0):
                        refineObj_temp = refineContainer.get_objects()[i - k]
                        max_level = max(max_level, refineObj_temp.levels[0])
                        if refineObj_temp.levels[0] <= refineObj.levels[1]:
                            break
                        k += 1
                    k = 1
                    while(i + k < len(refineContainer.get_objects())):
                        refineObj_temp = refineContainer.get_objects()[i + k]
                        max_level = max(max_level, refineObj_temp.levels[1])
                        if refineObj_temp.levels[1] <= refineObj.levels[1]:
                            break
                        k += 1
                    subtraction_value = self.lmax[d] - max_level
                else:
                    subtraction_value = 0
                if (refineObj.levels[1] <= max(levelvec[d] - subtraction_value, 1)):
                    indicesDim.append(refineObj.end)
                    if (next_refineObj is not None and self.is_child(refineObj.levels[0], refineObj.levels[1], next_refineObj.levels[0])) and not self.use_local_children:
                        children_indices_dim.append(self.get_node_info(refineObj.end, refineObj.levels[1], refineObj.start, refineObj.levels[0], next_refineObj.end, next_refineObj.levels[1], d))
                    if self.use_local_children:
                        indices_levelDim.append(refineObj.levels[1])
            if self.use_local_children:
                for i in range(1,len(indices_levelDim)-1):
                    if self.is_child(indices_levelDim[i-1], indices_levelDim[i], indices_levelDim[i+1]):
                        children_indices_dim.append((self.get_node_info(i, indicesDim, indices_levelDim, d)))
                #print(children_indices_dim, indices_levelDim)
            indicesList.append(indicesDim)
            children_indices.append(children_indices_dim)
        return indicesList, children_indices

    # returns if the coordinate refineObj.levels[1] is a child in the global refinement structure
    def is_child(self, level_left_point, level_point, level_right_point):
        return (level_left_point < level_point and level_right_point < level_point) and level_point > 1
        #return True
        if level_left_point < level_point or level_right_point < level_point:
            return True
        else:
            return False

    # This method calculates the left and right parent of a child. It might happen that a child has already a child
    # in one direction but it may not have one in both as it would not be considered to be a child anymore.
    def get_node_info(self, position, coords_dim, level_dim, d):
        child = coords_dim[position]
        level_child = level_dim[position]
        if self.equidistant:
            width = (self.b[d] - self.a[d]) / 2**level_child
            return NodeInfo(child, child - width, child + width, True, True, None,None)
        else:
            left_parent = None
            for i in reversed(range(position)):
                if level_dim[i] < level_child:
                    left_parent = coords_dim[i]
                    break
            assert left_parent is not None
            right_parent = None
            for i in (range(position+1, len(coords_dim))):
                if level_dim[i] < level_child:
                    right_parent = coords_dim[i]
                    break
            assert right_parent is not None
            return NodeInfo(child, left_parent, right_parent, True, True, None,None)

    # this method draws the 1D refinement of each dimension individually
    def draw_refinement(self, filename=None, markersize=10):  # update with meta container
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
            ax[d].plot(xValues, yValues, 'bo', markersize=markersize, color="black")
            ax[d].set_xlim([self.a[d], self.b[d]])
            ax[d].set_ylim([-0.1, 0.1])
            ax[d].set_yticks([])
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
        plt.show()
        return fig

    def init_evaluation_operation(self, areas):
        self.operation.initialize_refinement_container_dimension_wise(areas[0])

    def evaluate_operation_area(self, component_grid, area, additional_info=None):
        if self.grid.is_global():
            # get 1d coordinates of the grid points that define the grid; they are calculated based on the levelvector
            gridPointCoordsAsStripes, children_indices = self.get_point_coord_for_each_dim(component_grid.levelvector)

            # calculate the operation on the grid
            integral = self.operation.calculate_operation_dimension_wise(gridPointCoordsAsStripes, component_grid, self.a, self.b, self.refinements != 0 and not self.do_high_order and not self.grid.modified_basis)

            # compute the error estimates for further refining the Refinementobjects and therefore the future grid
            self.operation.compute_error_estimates_dimension_wise(gridPointCoordsAsStripes, children_indices, component_grid)

            # save the number of evaluations used per d-1 dimensional slice
            for d in range(self.dim):
                factor = component_grid.coefficient if self.grid.isNested() else 1
                self.evaluationCounts[d][component_grid.levelvector[d] - 1] += factor * np.prod([self.grid.numPoints[d2] if d2 != d else 1 for d2 in range(self.dim)])
            return np.prod(self.grid.numPoints)
        else:
            pass

    # This method computes additional values after the compution of the integrals for the current
    # refinement step is finished. This method is executed before the refinement process.
    def finalize_evaluation_operation(self, areas, evaluation_array):
        super().finalize_evaluation_operation(areas, evaluation_array)

        if self.version == 1:
            for d in range(self.dim):
                container_d = self.refinement.get_refinement_container_for_dim(d)
                for area in container_d.get_objects():
                    level = max(area.levels)
                    area.set_evaluations(np.sum(self.evaluationCounts[d][level-1:]))

    def initialize_refinement(self):
        initial_points = []
        for d in range(self.dim):
            initial_points.append(np.linspace(self.a[d], self.b[d], 2 ** 2 + 1))
        levels = [0, 2, 1, 2, 0]
        self.refinement = MetaRefinementContainer([RefinementContainer
                                                   ([RefinementObjectSingleDimension(initial_points[d][i],
                                                                                     initial_points[d][i + 1], d, self.dim, (levels[i], levels[i+1]),
                                                                                     self.lmax[d] - 2, dim_adaptive=self.dim_adaptive) for i in
                                                     range(2 ** 2)], d, self.errorEstimator) for d in
                                                   range(self.dim)])
        if self.dim_adaptive:
            self.combischeme.init_adaptive_combi_scheme(self.lmax[0], self.lmin[0])
        self.evaluationCounts = [np.zeros(self.lmax[d]) for d in range(self.dim)]
        if self.operation is not None:
            self.operation.init_dimension_wise(self.grid, self.grid_surplusses, self.f, self.refinement, self.lmin, self.lmax, self.a, self.b, self.version)


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
            for d in range(self.dim):
                self.lmax[d] += lmaxChange[d]
            if self.dim_adaptive:
                if self.print_output:
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
            #print("New scheme:")
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

