import abc
from scipy.interpolate import interpn
from sparseSpACE.Hierarchization import *
from typing import Callable, Tuple, Sequence

# This is the abstract interface of an integrator that integrates a given area specified by start for function f
# using numPoints many points per dimension
class IntegratorBase(object):
    @abc.abstractmethod
    def __call__(self, f: Callable[[Tuple[int, ...]], float], numPoints: Sequence[int], start: Sequence[float], end: Sequence[float]) -> Sequence[float]:
        pass

# This integrator computes the trapezoidal rule for the given interval without constructing the grid explicitly
class IntegratorTrapezoidalFast(IntegratorBase):
    def __call__(self, f: Callable[[Tuple[float, ...]], float], numPoints: Sequence[int], start: Sequence[float], end: Sequence[float]) -> Sequence[float]:
        dim = len(start)
        length = np.empty(dim)
        offsets = np.ones(dim, dtype=np.int64)
        spacing = np.empty(dim)
        gridsize = np.int64(1)
        for i in range(dim):
            length[i] = end[i] - start[i]
            spacing[i] = float(length[i]) / float(numPoints[i] - 1)
            gridsize *= np.int64(numPoints[i])
            if i != 0:
                offsets[i] = offsets[i - 1] * int(numPoints[i - 1])
        h_prod = np.prod(spacing)
        result = 0.0
        for i in range(gridsize):
            position = np.zeros(dim)
            rest = i
            factor = 0
            for d in reversed(list(range(dim))):
                position[d] = start[d] + int(rest / offsets[d]) * spacing[d]
                if int(rest / offsets[d]) == 0 or int(rest / offsets[d]) == numPoints[d] - 1:
                    factor += 1
                rest = rest % offsets[d]
            result += f(position) * 0.5 ** factor * h_prod
        del length
        del offsets
        del spacing
        return result


# This integrator computes the trapezoidal rule for the given interval by constructing the grid explicitly
# and applying iteratively 1D trapezoidal rules
class IntegratorGenerateGridTrapezoidal(IntegratorBase):
    def __call__(self, f: Callable[[Tuple[float, ...]], float], numPoints: Sequence[int], start: Sequence[float], end: Sequence[float]) -> Sequence[float]:
        dim = len(start)
        length = np.zeros(dim)
        offsets = np.ones(dim, dtype=np.int64)
        spacing = np.zeros(dim)
        gridsize = np.int64(1)
        for i in range(dim):
            length[i] = end[i] - start[i]
            spacing[i] = float(length[i]) / float(numPoints[i] - 1)
            gridsize *= np.int64(numPoints[i])
            if i != 0:
                offsets[i] = offsets[i - 1] * int(numPoints[i - 1])
        startTime = time.time()
        gridValues = np.zeros(gridsize)
        for i in range(gridsize):
            position = np.zeros(dim)
            rest = i
            for d in reversed(list(range(dim))):
                position[d] = start[d] + int(rest / offsets[d]) * spacing[d]
                rest = rest % offsets[d]
            gridValues[i] = f(position)
        endTime = time.time()
        startTime = time.time()
        currentSliceSize = gridsize
        for d in reversed(list(range(dim))):
            currentSliceSize = int(currentSliceSize / int(numPoints[d]))
            for i in range(currentSliceSize):
                lineValues = np.zeros(int(numPoints[d]))
                for j in range(int(numPoints[d])):
                    lineValues[j] = gridValues[i + j * offsets[d]].copy()
                gridValues[i] = np.trapz(lineValues, dx=spacing[d])
                del lineValues
        endTime = time.time()
        result = gridValues[0].copy()
        del gridValues
        gc.collect()
        return result


# This integrator computes the integral of an arbitrary grid from the Grid class
# using the predefined interfaces and weights. The grid is not explicitly constructed.
class IntegratorArbitraryGrid(IntegratorBase):
    def __init__(self, grid):
        self.grid = grid

    def __call__(self, f: Callable[[Tuple[float, ...]], float], numPoints: Sequence[int], start: Sequence[float], end: Sequence[float]) -> Sequence[float]:
        dim = len(start)
        offsets = np.ones(dim, dtype=np.int64)
        gridsize = np.int64(1)
        for i in range(dim):
            gridsize *= np.int64(numPoints[i])
            if i != 0:
                offsets[i] = offsets[i - 1] * int(numPoints[i - 1])
        result = 0.0
        for i in range(gridsize):
            indexvector = np.empty(dim, dtype=int)
            rest = i
            for d in range(dim - 1, -1, -1):
                indexvector[d] = int(rest / offsets[d])
                rest = rest % offsets[d]
            result += self.integrate_point(f, indexvector)
        del offsets
        return result

    def integrate_point(self, f, indexvector):
        weight = self.grid.getWeight(indexvector)
        if weight == 0:
            return 0.0
        position = self.grid.getCoordinate(indexvector)
        return f(position) * weight


# This integrator computes the integral of an arbitrary grid from the Grid class
# using the predefined interfaces and weights. The grid is explicitly constructed and efficiently evaluated using numpy.
# If the function is a vector valued function we evaluate a matrix vector product with the weights, i.e. integrate
# each component individually.
class IntegratorArbitraryGridScalarProduct(IntegratorBase):
    def __init__(self, grid):
        self.grid = grid #type: Grid

    def __call__(self, f: Function, numPoints: Sequence[int], start: Sequence[float], end: Sequence[float]) -> Sequence[float]:
        points, weights = self.grid.get_points_and_weights()
        f_values = f(points)

        if len(f_values) == 0:
            assert len(points) == 0
            assert len(weights) == 0
            return 0.0
        else:
            return np.inner(f_values.T, weights)

'''
#This integrator computes the integral of an arbitrary grid from the Grid class
#using the predefined interfaces and weights. The grid is not explicitly constructed.
#GridPointsAreExcluded
class IntegratorArbitraryGridNoBoundary(IntegratorBase):
    def __init__(self,grid):
        self.grid = grid

    def __call__(self,f,numPoints,start,end):
        dim = len(start)
        offsets = np.ones(dim,dtype=np.int64)
        gridsize = np.int64(1)
        numPointsEval = list(numPoints)
        indexOffset = np.zeros(dim,dtype=np.int64)
        for i in range(dim):
            if start[i]==0 :
                numPointsEval[i] -= 1
                indexOffset[i] = 1
            if end[i] == 1:
                numPointsEval[i] -= 1
            gridsize *= np.int64(numPointsEval[i])
            if i != 0:
                offsets[i] = offsets[i-1] * int(numPointsEval[i-1])
        result = 0.0
        for i in range(gridsize):
            indexvector = np.zeros(dim,dtype=int)
            rest = i
            for d in reversed(list(range(dim))):
                indexvector[d] = int(rest / offsets[d]  + indexOffset[d])
                rest = rest % offsets[d]
            position = self.grid.getCoordinate(indexvector)
            result += f(position) * self.grid.getWeight(indexvector)
        del offsets
        return result
'''


# This is a helper method used in the single dimension method.
# It interpolates the grid to a finer grid and computes partial integrals for the subareas
def integrateVariableStartNumpyArbitraryDimAndInterpolate(f, numPoints, start, end, numberOfGridsContained):
    dim = len(start)
    length = np.zeros(dim)
    offsets = np.ones(dim, dtype=np.int64)
    extendedOffsets = np.ones(dim, dtype=np.int64)  # extended to interpolated Array

    spacing = np.zeros(dim)
    extendedSpacing = np.zeros(dim)
    gridsize = np.int64(1)
    for i in range(dim):
        length[i] = end[i] - start[i]
        spacing[i] = float(length[i]) / float(numPoints[i] - 1)
        extendedSpacing[i] = spacing[i] / numberOfGridsContained[i]
        gridsize *= np.int64(numPoints[i])
        if i != 0:
            offsets[i] = offsets[i - 1] * int(numPoints[i - 1])
            extendedOffsets[i] = offsets[i - 1] * (((int(numPoints[i - 1]) - 1) * numberOfGridsContained[i - 1]) + 1)
    startTime = time.time()

    gridValues = np.zeros(gridsize)
    for i in range(gridsize):
        position = np.zeros(dim)
        rest = i
        for d in reversed(list(range(dim))):
            position[d] = start[d] + int(rest / offsets[d]) * spacing[d]
            rest = rest % offsets[d]
        gridValues[i] = f(position)
    # number of Points of extended Array
    extendedNumPoints = [(numPoints[d] - 1) * numberOfGridsContained[d] + 1 for d in range(dim)]
    # corresponding Points
    extendedPoints = [np.linspace(start[d], end[d], extendedNumPoints[d]) for d in range(dim)]
    interp_mesh = np.array(np.meshgrid(*extendedPoints))
    inter_points = [g.ravel() for g in interp_mesh]  # flattened point mesh
    pointCoordinates = [np.linspace(start[d], end[d], numPoints[d]) for d in range(dim)]
    gridValuesInterpolated = interpn(pointCoordinates, gridValues.reshape(*reversed(numPoints)).transpose(),
                                     list(zip(*inter_points)))
    endTime = time.time()
    # calculate grid combinations
    indices = [list(range(numGrids)) for numGrids in numberOfGridsContained]
    # print indices
    gridCoord = list(zip(*[g.ravel() for g in np.meshgrid(*indices)]))
    # calculate point coordinates
    indices = [list(range(numPerDim)) for numPerDim in numPoints]
    # print indices
    points = list(zip(*[g.ravel() for g in np.meshgrid(*indices)]))
    gridIntegrals = []
    startoffsetGrid = np.zeros(dim)
    endoffsetGrid = np.zeros(dim)
    startTime = time.time()
    for g in range(np.prod(numberOfGridsContained)):
        offsetGrid = np.zeros(dim)
        for d in range(dim):
            offsetGrid[d] = (numPoints[d] - 1) * float(gridCoord[g][d])
            startoffsetGrid[d] = start[d] + float(gridCoord[g][d]) / float(numberOfGridsContained[d]) * length[d]
            endoffsetGrid[d] = start[d] + float(gridCoord[g][d] + 1) / float(numberOfGridsContained[d]) * length[d]
        # copy part of interpolated grid to gridsize
        for i in range(gridsize):
            position = np.inner(extendedOffsets, np.array(points[i]) + offsetGrid)
            gridValues[i] = gridValuesInterpolated[int(position)]
        # calculates integral for current subGrid
        currentSliceSize = gridsize
        for d in reversed(list(range(dim))):
            currentSliceSize = int(currentSliceSize / int(numPoints[d]))
            for i in range(currentSliceSize):
                lineValues = np.zeros(int(numPoints[d]))
                for j in range(int(numPoints[d])):
                    lineValues[j] = gridValues[i + j * offsets[d]]
                gridValues[i] = np.trapz(lineValues, dx=extendedSpacing[d])
        gridIntegrals.append((gridValues[0], np.array(startoffsetGrid), np.array(endoffsetGrid)))
    endTime = time.time()
    return gridIntegrals


class IntegratorHierarchicalBasisFunctions(IntegratorBase):
    def __init__(self, grid):
        self.grid = grid
        self.hierarchization = HierarchizationLSG(grid)
        self.surplus_values = None

    def __call__(self, f: Callable[[Tuple[float, ...]], float], numPoints: Sequence[int], start: Sequence[float], end: Sequence[float]) -> Sequence[float]:
        output_dim = f.output_length()
        grid_values = np.empty((output_dim, np.prod(numPoints)))
        points = self.grid.getPoints()
        for i, point in enumerate(points):
            v = f(point)
            assert len(v) == output_dim, "The Function returned a wrong output length"
            grid_values[:, i] = v
        self.surplus_values = self.hierarchization(grid_values, numPoints, self.grid)
        weights = self.grid.get_weights()
        #print(sum(weights), np.prod(np.array(end) - np.array(start)), start,end, weights, self.grid.weights)
        #if not isclose(sum(weights), np.prod(np.array(end) - np.array(start))):
        #    assert False
        #print(np.inner(self.surplus_values, weights))
        if len(self.surplus_values) == 0:
            return 0.0
        else:
            return np.inner(self.surplus_values, weights)

    def get_surplusses(self):
        return self.surplus_values

class IntegratorHierarchical(IntegratorBase):
    def __call__(self, f: Callable[[Tuple[float, ...]], float], numPoints: Sequence[int], start: Sequence[float], end: Sequence[float]) -> Sequence[float]:
        print("Normal Call in hierarchicalIntegror... does not work!")
        assert False
        # NotImplementedError()

    def integrate_hierarchical(self, f, gridPointCoordsForEachDimension, start, end, widthPerDimension, metaRefineContainer, levelVec, lmin, lmax):
        self.dim = len(start)
        self.start = start
        self.end = end
        self.metaRefineContainer = metaRefineContainer
        self.dictChildren = metaRefineContainer.dictChildren
        # print("dictChildren!", self.dictChildren)
        self.dictCoordinate = metaRefineContainer.dictCoordinate
        # print("dictCoordinate", self.dictCoordinate)
        self.depthLevelVector = metaRefineContainer.depthLevelVector
        # print("depthLevelVector", self.depthLevelVector)
        self.widthPerDimension = widthPerDimension
        self.f = f
        # coordiantes of points, but sorted as dimensional stripes, so like (0, 0.5, 1),(0, 1) for example
        self.gridPointCoordsForEachDimension = gridPointCoordsForEachDimension
        # for statistics:
        self.evaluations = 0
        # the values of the gridpoints
        self.gridPoints = self.init_grid_point_values()
        # same as gridPointVolumes, but only the volumes of the children points are set to non-zero values (if they are non-zero)
        self.gridPointVolumeChildren = np.zeros(len(self.gridPoints))
        # determines the level of each point in each dimension
        self.gridPointLevels = []
        for i in range(0, len(self.gridPoints)):
            self.gridPointLevels.append([])
        # the different integrators, each calculates stripes for one specific dimension
        self.integratorsForOneDim = []

        # loop over all this_dim and construct stripes and build them
        for d in range(0, self.dim):
            newGrid = IntegratorHierarchicalForOneDim(d, self.depthLevelVector[d], self.dictChildren[d])
            self.integratorsForOneDim.append(newGrid)
            self.construct_stripes_for_one_dim(d)
        # calculate results
        # setting the volumes in the refinement objects:
        integral = self.set_integral_and_error(lmax, lmin, levelVec)
        # print("GridPointLevels:", self.gridPointLevels)
        # print("Evaluations:", self.evaluations)
        return integral, None, self.evaluations, self.metaRefineContainer

    def init_grid_point_values(self):
        gridPointValues = np.zeros(len(self.dictCoordinate))
        for key in self.dictCoordinate:
            value = self.f.eval(key)
            # self.evaluations += 1
            index = self.dictCoordinate[key]
            gridPointValues[index] = value
        return gridPointValues

    def get_point_from_grid_coord(self, coord):
        index = self.dictCoordinate[coord]
        value = self.gridPoints[index]
        return [int(index), value]

    def construct_stripes_for_one_dim(self, thisDim):
        # print("--------------")
        # print("BuildHierarchicalStripes for Dim", thisDim)
        # print("GridPoints before", self.gridPoints)
        stripe = np.zeros((self.widthPerDimension[thisDim]))  # ,) * self.this_dim)
        stripeIndices = np.zeros((self.widthPerDimension[thisDim]), dtype=np.int64)

        all_other_coord_combinations = self.calc_all_other_coord(thisDim)
        # loop over every OTHER dimension than "thisDim"
        for other_coord in all_other_coord_combinations:
            for offsetIndex in range(0, self.widthPerDimension[thisDim]):
                offset = self.gridPointCoordsForEachDimension[thisDim][offsetIndex]
                coord = np.zeros(self.dim)
                for other_dim in range(0, self.dim):
                    # print("coord zuweisung", other_coord, "\t offset", offset, "curCoord", coord)
                    if (thisDim == other_dim):
                        coord[other_dim] = offset
                    else:
                        coord[other_dim] = other_coord[other_dim]
                coord = tuple(coord)

                indice, value = self.get_point_from_grid_coord(coord)
                self.evaluations += 1
                stripe[offsetIndex] = value
                # print("coord", coord, "\tvalue", point[1])
                stripeIndices[offsetIndex] = indice
                # buildStripe is a [stripe, leaves]
                # print("d", d, "stripe", stripe)
            buildStripe, leaves, levels = self.integratorsForOneDim[thisDim].build_one_stripe(stripe)
            for point in levels:
                index = stripeIndices[point[0]]
                level = point[1]
                self.gridPointLevels[index].append(level)
            for leaf in leaves:
                # pagodaVolume = self.calc_error_estimation_for_one_point(leaf[0], leaf[1])[0]
                gridPointIndex = stripeIndices[leaf[2]]
                self.gridPointVolumeChildren[gridPointIndex] += 1
                # print("leaf index", gridPointIndex, "len(gridPointVolumeChildren", len(self.gridPointVolumeChildren))
            # "in-place" principle done here!
            for pnt in range(0, len(buildStripe)):
                self.gridPoints[stripeIndices[pnt]] = buildStripe[pnt]
        # print("gridPointLeafCounterPerPoint", self.gridPointVolumeChildren)
        # print("GridPoints after", thisDim, " equals", self.gridPoints)
        return True

    def calc_all_other_coord(self, thisDim):
        indicesList = []
        for d in range(0, self.dim):
            if (d == thisDim):
                indicesList.append([0])
            else :
                indicesList.append(self.gridPointCoordsForEachDimension[d])
        allPoints = list(set(zip(*[g.ravel() for g in np.meshgrid(*indicesList)])))
        # print("allPointsOther", allPoints)
        return allPoints

    def set_integral_and_error(self, lmax, lmin, levelVec):
        # for testing purposes
        # print("set_integral_and_error")
        integral = 0
        # calculate whole integral:
        for point_index in range(0, len(self.gridPoints)):
            point = self.gridPoints[point_index]
            pagoda_volume = self.calc_error_estimation_for_one_point(point_index, point)
            if (self.gridPointVolumeChildren[point_index] == self.dim):
                self.gridPointVolumeChildren[point_index] = pagoda_volume
            else:
                self.gridPointVolumeChildren[point_index] = 0
            integral += pagoda_volume
        numSubDiagonal = (lmax[0] + (self.dim - 1) * lmin[0]) - np.sum(levelVec)
        # print("GridPointVolumeChildren:", self.gridPointVolumeChildren)
        self.set_volume_of_children(numSubDiagonal)
        return integral

    def calc_error_estimation_for_one_point(self, index, surplus):
        level_sum = 0
        for level in self.gridPointLevels[index]:
            level_sum += max(level, 1)
        # print("point", index, " has level_sum", level_sum)
        volume = 2**(-1 * level_sum)
        pagoda_volume = surplus * volume
        return pagoda_volume

    def set_volume_of_children(self, numSubDiagonal):
        # setting up structure of refinementObjects (so i do not have to add up later but rather set the total volume right away
        containerStructure, fullGridPointCoordsForEachDimension = self.buildContainerStructure(self.metaRefineContainer)
        # print("containerStructure", containerStructure)
        # sind die fullgridPointcoords immer aufsteigend sortiert?
        # print("fullGridPointCoordsForEachDimension", fullGridPointCoordsForEachDimension)
        # print("gridPointVolumeChildren", self.gridPointVolumeChildren)
        # print("gridPointLevels", self.gridPointLevels)
        if (numSubDiagonal == 0):
            # setting the volumes in the refinement objects:
            for key in self.dictCoordinate:
                key_index = self.dictCoordinate[key]
                if (self.gridPointVolumeChildren[key_index] != 0):
                    for d in range(0, self.dim):
                        coord = key[d]
                        level = self.gridPointLevels[key_index][d]
                        half_width = (self.end[d] - self.start[d]) / (2**int(level))
                        min_coord = max(self.start[d], coord - half_width)
                        max_coord = min(self.end[d], coord + half_width)
                        # print("setVolume, level", level, "\tminCoord", min_coord, "\tMaxCoord", max_coord)
                        # print("key", key, "\tkey_index", key_index, "\tcoord", coord)
                        for g in range(0, len(fullGridPointCoordsForEachDimension[d])-1):
                            comparison_coord = fullGridPointCoordsForEachDimension[d][g]
                            # print("comparisonCoord", comparison_coord, "\tGridCoord", self.gridPointCoordsForEachDimension[d])
                            if (comparison_coord < max_coord and comparison_coord >= min_coord):
                                target_obj = self.metaRefineContainer.get_refinement_container_for_dim(d).get_object(g)
                                target_start = target_obj.start
                                target_end = target_obj.end
                                target_width = target_end - target_start
                                factor = target_width / (2*half_width)
                                assert(target_width <= 2*half_width)
                                newError = (self.gridPointVolumeChildren[key_index] * factor)
                                # if (newError > containerStructure[d][g]):
                                    # print("new error")
                                    # containerStructure[d][g] = (self.gridPointVolumeChildren[key_index] * factor)
                                containerStructure[d][g] += (self.gridPointVolumeChildren[key_index] * factor)
                                # TODO fehler überschreiben, wenn größer als bisheriger (also max fehler von allen gittern!)
                                # print("added at", d, g, "volume:", self.gridPointVolumeChildren[key_index])
                                # print("containerStructure", containerStructure)
            # print("containerStructure", containerStructure)
            for d in range(self.dim):
                for offset in range(len(containerStructure[d])):
                    # print("add_volume_to", d, "id", offset, "volume", containerStructure[d][offset])
                    # container = self.metaRefineContainer.refinementContainers[d]
                    # error = containerStructure[d][offset]
                    # if abs(error) > abs(container.get_object(offset).volume):
                    #     container.get_object(offset).volume = error
                    self.metaRefineContainer.refinementContainers[d].add_volume_of_children(offset, containerStructure[d][offset], self.f)
        # else:
            # print("No Errors because it is not one of the current grids")

    def buildContainerStructure(self, metaRefineContainer):
        containerStructure = []
        for d in range(0, self.dim):
            width = metaRefineContainer.get_refinement_container_for_dim(d).size()
            containerStructure.append(np.zeros(width))
        # buidling the full gridPointCoordsForEachDimension:
        # get a list of all coordinates for every this_dim (so (0, 1), (0, 0.5, 1) for example)
        indicesList = []
        for d in range(0, self.dim):
            refineContainer = metaRefineContainer.get_refinement_container_for_dim(d)
            indices = []
            indices.append(refineContainer.get_objects()[0].start)
            for refineObj in refineContainer.get_objects():
                # print("levelvec", levelvec)
                indices.append(refineObj.end)
            indicesList.append(indices)
        return containerStructure, indicesList

    def plot_structure(self):
        print("--------------")
        print("Superclass Grid with this_dim", self.dim)
        print("GridPoints", self.gridPoints)
        print("dictCoordinate", self.dictCoordinate)
        print("dictChildren", self.dictChildren)
        print("depthLevelVector", self.depthLevelVector)
        print("--------------")
        for obj in self.integratorsForOneDim:
            obj.plot_integrator_for_one_dim()


class IntegratorHierarchicalForOneDim:
    def __init__(self, thisDim, levelVector, dictChildren):
        # the dimension this object is for
        self.thisDim = thisDim
        # Dictionary, mapping positions to the respective Children
        self.dictChildren = dictChildren
        # the level Vector for this this_dim
        self.levelVector = levelVector
        # self.plot_integrator_for_one_dim()

    def plot_integrator_for_one_dim(self):
        print ("--------------")
        print ("Integrator for Dim:", self.thisDim)
        print ("dictChildren", self.dictChildren)
        print ("depthLevelVector", self.levelVector)

    # the stripes are sorted after the indices of other data (like dictChildren), but has the values in it
    def build_one_stripe(self, stripe):
        # stripe is a list with point values
        assert(len(stripe) == len(self.dictChildren))
        leaves = []
        levels = []
        for level in range(len(self.levelVector)-2, -1, -1):
            for point in self.levelVector[level]:
                levels.append([point, level])
                for child in self.dictChildren[point]:
                    # in-place update
                    stripe[child] = self.update_point(stripe[child], stripe[point])
            if (self.dictChildren[point] == [0]):
                # when having no children, it is a leaf
                leaves.append([stripe[point], level, point])
        # adding the obvious leaves (from deepest level)
        for leaf in self.levelVector[len(self.levelVector)-1]:
            levels.append([leaf, len(self.levelVector)-1])
            leaves.append([stripe[leaf], len(self.levelVector)-1, leaf])
        # print ("Leaves:", leaves)
        return [stripe, leaves, levels]

    # point and parent are both grid point values
    def update_point(self, point, parent):
        point -= 0.5 * parent
        return point
