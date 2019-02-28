from spatiallyAdaptiveBase import *
from Grid import *

def sortToRefinePosition(elem):
    # sort by depth
    return elem[1]


class SpatiallyAdaptiveSingleDimensions2(SpatiallyAdaptivBase):
    def __init__(self, a, b):
        self.grid = GlobalTrapezoidalGrid(a, b, boundary=True)
        SpatiallyAdaptivBase.__init__(self, a, b, self.grid)

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
            k = 0
            for i in range(len(refineContainer.get_objects())):
                refineObj = refineContainer.get_objects()[i]
                if i + 1 < len(refineContainer.get_objects()):
                    next_refineObj = refineContainer.get_objects()[i + 1]
                else:
                    next_refineObj = None
                if (refineObj.levels[1] <= levelvec[d]):
                    k += 1
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
        assert right_parent - child == child - left_parent
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
    def evaluate_area(self, f, area, levelvec):
        if(self.grid.is_global() == True):
            gridPointCoordsAsStripes, children_indices = self.get_point_coord_for_each_dim(levelvec)
            start = self.a
            end = self.b
            self.grid.set_grid(gridPointCoordsAsStripes)
            integral = self.grid.integrator(f, self.grid.numPoints, start, end)
            num_sub_diagonal = (self.lmax[0] + self.dim - 1) - np.sum(levelvec)
            if num_sub_diagonal == 0:
                self.calculate_surplusses(gridPointCoordsAsStripes, children_indices)
            return integral, None, np.prod(self.grid.numPoints)
        else:
            pass

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
        assert right_parent - child == child - left_parent
        points_left_parent = list(zip(*[g.ravel() for g in np.meshgrid(*[grid_points[d2] if d != d2 else [left_parent] for d2 in range(self.dim)])]))
        points_right_parent = list(zip(*[g.ravel() for g in np.meshgrid(*[grid_points[d2] if d != d2 else [right_parent] for d2 in range(self.dim)])]))
        points_children = list(zip(*[g.ravel() for g in np.meshgrid(*[grid_points[d2] if d != d2 else [child] for d2 in range(self.dim)])]))
        indices = list(zip(*[g.ravel() for g in np.meshgrid(*[range(len(grid_points[d2])) if d != d2 else None for d2 in range(self.dim)])]))
        for i in range(len(points_children)):
            index = indices[i]
            factor = np.prod([self.grid.weights[d2][index[d2]] if d2 != d else 1 for d2 in range(self.dim)])
            volume += factor * abs(self.f(points_children[i]) - 0.5 * (self.f(points_left_parent[i]) + self.f(points_right_parent[i]))) * (right_parent - child)
        evaluations = 3 * len(points_right_parent)
        return max(abs(volume)), evaluations

    def initialize_refinement(self):
        initial_points = []
        for d in range(self.dim):
            initial_points.append(np.linspace(self.a[d], self.b[d], 2 ** 1 + 1))
        self.refinement = MetaRefinementContainer([RefinementContainer
                                                   ([RefinementObjectSingleDimension(initial_points[d][i],
                                                                                     initial_points[d][i + 1], d, self.dim, (i % 2, (i+1) % 2),
                                                                                     self.lmax[d] - 1) for i in
                                                     range(2 ** 1)], d, self.errorEstimator) for d in
                                                   range(self.dim)])

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
            print("New scheme")
            self.scheme = self.combischeme.getCombiScheme(self.lmin[0], self.lmax[0], do_print=False)
            return False
        return False

    def refinement_postprocessing(self):
        self.refinement.apply_remove(sort=True)
        self.refinement.refinement_postprocessing()
        self.refinement.reinit_new_objects()

class NodeInfo():
    def __init__(self, child, left_parent, right_parent, has_left_child, has_right_child, left_refinement_object, right_refinement_object):
        self.child = child
        self.left_parent = left_parent
        self.right_parent = right_parent
        self.has_left_child = has_left_child
        self.has_right_child = has_right_child
        self.left_refinement_object = left_refinement_object
        self.right_refinement_object = right_refinement_object

