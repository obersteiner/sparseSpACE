from spatiallyAdaptiveBase import *
from Grid import *

def sortToRefinePosition(elem):
    # sort by depth
    return elem[1]


class SpatiallyAdaptiveSingleDimensions(SpatiallyAdaptivBase):
    def __init__(self, a, b, grid):
        self.grid = GlobalHierarchizationGrid(a, b, boundary=True)
        SpatiallyAdaptivBase.__init__(self, a, b, self.grid)
        self.errorEstimator = ErrorCalculatorSingleDimVolumeGuided()
        self.useToleranceAsRefineCond = False

    def coarsen_grid(self, area, levelvec):
        pass
        '''
        #a = self.a[0]
        #b = self.b[0]
        containedInGrid = True #the areas might be very fine -> it might not be contained in a coarse grid
        startOfGridRange = True #this indicates that the area starts at a gridpoint on the grid
        startArea = np.zeros(self.this_dim)
        endArea = np.zeros(self.this_dim)
        for d in range(self.this_dim):
            startArea[d] = self.refinement[d][area[d]][0]
            endArea[d] = self.refinement[d][area[d]][1]
            width = (endArea[d] - startArea[d])
            if (self.b[d]-self.a[d])/width > 2**levelvec[d]: #this means area is too fine
                containedInGrid = False
                #check if area is at start of points in the grid -> extend area to grid resolution
                #this avoids that we extend areas more than once
                widthCoarse = (self.b[d]-self.a[d])/float(2**levelvec[d])
                if int(startArea[d]/widthCoarse) != startArea[d] / widthCoarse:
                    startOfGridRange = False
                else:
                    endArea[d] = startArea[d] + widthCoarse
        if(not containedInGrid and not startOfGridRange): #this area is not considered in this grid
            return None, startArea, endArea
        #calculate the number of points in this area
        levelCoarse = np.zeros(self.this_dim,dtype=int)
        for d in range(self.this_dim):
            coarseningFactor = self.refinement[d][area[d]][2]
            levelCoarse[d] = max(int(levelvec[d]  +  math.log(((endArea[d] - startArea[d]) / (self.b[d]-self.a[d])*1.0), 2)   - coarseningFactor), 0)
        return levelCoarse, startArea, endArea
        '''

    # returns the points coordinates of a single component grid with refinement
    def get_points_all_dim(self, levelvec, numSubDiagonal):
        indicesList = self.get_point_coord_for_each_dim(levelvec)
        # print ("------------------\nIndices", indicesList)
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
        for d in range(0, self.dim):
            refineContainer = refinement.get_refinement_container_for_dim(d)
            indices = []
            indices.append(refineContainer.get_objects()[0].start)
            for refineObj in refineContainer.get_objects():
                # print("levelvec", levelvec)
                if (refineObj.levels[1] <= levelvec[d]):
                    indices.append(refineObj.end)
            indicesList.append(indices)
        return indicesList

    # calculates number of points for each dimension, returning it as whole package
    def get_num_points_all_dim(self, levelvec):
        widthPerDimension = np.zeros(self.dim, dtype=np.int64)
        point_coord_for_each_dim = self.get_point_coord_for_each_dim(levelvec)
        for d in range(0, self.dim):
            widthPerDimension[d] = len(point_coord_for_each_dim[d])
        # print("widthPerDimension", widthPerDimension)
        return widthPerDimension

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
        # print("---------------\nEvaluate area!", levelvec)
        if(self.grid.is_global() == True):
            gridPointCoordsAsStripes = self.get_point_coord_for_each_dim(component_grid.levelvectir)
            start = self.a
            end = self.b
            dim = self.dim
            widthPerDimension = self.get_num_points_all_dim(component_grid.levelvector)
            # initializes grid-specific refinement information
            self.init_grid_specific_info(component_grid.levelvector, widthPerDimension)

            integral, partial_integral, evaluations, metaRefineContainer = self.grid.integrator.integrate_hierarchical(f, gridPointCoordsAsStripes, start, end, widthPerDimension, self.refinement, component_grid.levelvector, self.lmin, self.lmax)
            self.refinement = metaRefineContainer
            return integral, partial_integral, evaluations
        else:
            pass
        '''
        levelCoarse, start, end = self.coarsenGrid(area,levelvec)
        if(levelCoarse == None):
            #return default value to indicate that we do not compute this area
            return -2**30, [], 0
        startArea = np.zeros(self.this_dim)
        endArea = np.zeros(self.this_dim)
        for d in range(self.this_dim):
            startArea[d] = self.refinement[d][area[d]][0]
            endArea[d] = self.refinement[d][area[d]][1]
        width = endArea - startArea
        partialIntegral = self.grid.integrate(f,levelCoarse,start,end) #calculate regular integral
        partialIntegrals = [(partialIntegral, start, end)]
        numGridsOfRefinement = np.ones(self.this_dim)
        #calculate the number of grids in each dimension that we need to interpolate for the error estimator
        if(not(np.equal(np.array(end)-np.array(start),np.array(width)).all())):
            numGridsOfRefinement = [int((end[d] - start[d]) / self.finestWidth )for d in range(self.this_dim)]
        #calculate the integral of the interpolated partial grids
        if(np.prod(numGridsOfRefinement) > 1):
            self.grid.setCurrentArea(start,end,levelvec)
            partialIntegrals = integrateVariableStartNumpyArbitraryDimAndInterpolate(f,self.grid.levelToNumPoints(levelCoarse),start,end,numGridsOfRefinement)
        return partialIntegral, partialIntegrals, np.prod(self.grid.levelToNumPoints(levelCoarse))

        '''

    # calculate the position of the partial integral in the variance array
    def get_position(self, partialIntegralInfo, dim):
        refinement_array = self.refinement
        start_partial_integral = partialIntegralInfo[1]
        end_partial_integral = partialIntegralInfo[2]
        positions = []
        for d in range(dim):
            position_dim = []
            for i in range(len(refinement_array[d])):
                r = refinement_array[d][i]
                start_ref = r[0]
                end_ref = r[1]
                if start_partial_integral[d] >= start_ref and end_partial_integral[d] <= end_ref:
                    position_dim.append(i)
            positions.append(position_dim)
        return positions

    def initialize_refinement(self):
        # toDo self, start, end, this_dim, coarseningLevel = 0
        # RefinementObjectSingleDimension <- what is this for?
        initial_points = []
        for d in range(self.dim):
            initial_points.append(np.linspace(self.a[d], self.b[d], 2 ** 1 + 1))
        self.refinement = MetaRefinementContainer([RefinementContainer
                                                   ([RefinementObjectSingleDimension(initial_points[d][i],
                                                                                     initial_points[d][i + 1], d, self.dim, (i % 2, (i+1) % 2),
                                                                                     self.lmax[d] - 1) for i in
                                                     range(2 ** 1)], d, self.errorEstimator) for d in
                                                   range(self.dim)])
        # self.finestWidth = (initial_points[0][1]-initial_points[0][0])/(self.b[0] - self.a[0])

    def init_grid_specific_info(self, levelVec, widthPerDimension):
        # print("init_grid_specific_info", levelVec)
        toRefinePosition = []
        lastIndices = []
        for d in range(0, self.dim):
            toRefinePosition.append([])
            container = self.refinement.refinementContainers[d]
            curLevelVec = levelVec[d]

            offset = -1 # because start object is always auto integrated
            for obj_index in range(0, container.size()):
                obj = container.get_object(obj_index)
                # if (obj.start != self.a[d] and obj.end != self.b[d] and obj.start != (self.b[d] - self.a[d] / 2) and (max(obj.levels) <= curLevelVec)):
                # if (max(obj.levels) <= curLevelVec):
                if (obj.start != self.a[d] and (obj.levels[0] <= curLevelVec)):
                    toRefinePosition[d].append(((obj_index - offset), obj.levels[0]))
                else:
                    offset += 1

            newLastIndex = container.size() - offset
            lastIndices.append(newLastIndex)

            if len(toRefinePosition[d]) > 0:
                toRefinePosition[d] = sorted(toRefinePosition[d], key=sortToRefinePosition)
        # print("toRefinePosition", toRefinePosition)

        dictChildren = self.init_dictChildren(levelVec, toRefinePosition, lastIndices)
        # print("dictChildren", dictChildren)
        depthLevelVector = self.init_depthLevelVector(levelVec, toRefinePosition, lastIndices)
        dictCoordinate = self.init_dictCoordinate(levelVec)

        self.refinement.set_dictChildren(dictChildren)
        self.refinement.set_dictCoordinate(dictCoordinate)
        self.refinement.set_depthLevelVector(depthLevelVector)

    def init_dictChildren(self, levelVec, toRefinePosition, lastIndices):
        # print("---------------\ninit_dictChildren")
        allDicts = []
        for d in range(0, self.dim):
            allDicts.append({0: [], lastIndices[d]: []})
            # toRefinePosition is sorted after the depth of the points
            refineObjectsInDim = toRefinePosition[d]

            for refine_index in range(0, len(refineObjectsInDim)):
                refine_obj = refineObjectsInDim[refine_index]
                depth = refine_obj[1] # always depth at start coord
                obj_index = refine_obj[0]

                allDicts[d][obj_index] = []
                neighbours = self.find_neighbours_in_dict(obj_index, allDicts[d], len(refineObjectsInDim))
                allDicts[d][neighbours[0]].append(obj_index)
                allDicts[d][neighbours[1]].append(obj_index)
        return allDicts

    def find_neighbours_in_dict(self, index, dictChildren, counter):
        # print("dictChildre", dictChildren)
        # print("index", index)
        defaultLeft = -1
        defaultRight = 100 + counter * 2
        leftNeighbourIndex = defaultLeft
        rightNeighbourIndex = defaultRight
        for key in dictChildren:
            if key < index and key > leftNeighbourIndex:
                leftNeighbourIndex = key
            if key > index and key < rightNeighbourIndex:
                rightNeighbourIndex = key
        if (leftNeighbourIndex == defaultLeft or rightNeighbourIndex == defaultRight):
            print("ERROR in finding neighbours!!!")
        else:
            return (leftNeighbourIndex, rightNeighbourIndex)

    def init_depthLevelVector(self, levelVec, toRefinePosition, lastIndices):
        allVectors = []
        for d in range(0, self.dim):
            allVectors.append([])
            allVectors[d].append([])
            allVectors[d].append([])
            allVectors[d][0].append(0)
            allVectors[d][0].append(lastIndices[d])
            # toRefinePosition is sorted after the depth of the points
            refineObjectsInDim = toRefinePosition[d]

            for refine_index in range(0, len(refineObjectsInDim)):
                refine_obj = refineObjectsInDim[refine_index]
                depth = refine_obj[1] # always depth at start coord
                obj_index = refine_obj[0]

                if (depth == len(allVectors[d])):
                    # new bucket
                    allVectors[d].append([obj_index])
                else:
                    allVectors[d][depth].append(obj_index)
        return allVectors

    def init_dictCoordinate(self, levelVec):
        # TODO intialize with ints not floats?
        pointCoords = self.get_points_all_dim(levelVec, 0) # todo argument not used at the moment
        dictCoordinate = {}
        index = 0
        for point in pointCoords:
            dictCoordinate[point] = index
            index += 1
        # can stay as it is (points get removed in self.get_points_all_dim anyway)
        # print("dictCoordinate after INIT:", dictCoordinate)
        return dictCoordinate

    # for testing purposes!
    def set_refinement(self, refinement):
        self.refinement = refinement

    def get_areas(self):
        if (self.grid.is_global() == True):
            return [self.refinement]
        # get a list of lists which contains range(refinements[d]) for each dimension d where the refinements[d] are the number of subintervals in this dimension
        indices = [list(range(len(refineDim))) for refineDim in self.refinement.get_new_objects()]
        # this command creates tuples of size this_dim of all combinations of indices (e.g. this_dim = 2 indices = ([0,1],[0,1,2,3]) -> areas = [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3)] )
        return list(zip(*[g.ravel() for g in np.meshgrid(*indices)]))

    def get_new_areas(self):
        return self.get_areas()

    def prepare_refinement(self):
        pass
        # this is done in meta container
        '''
        self.popArray = [[] for d in range(self.this_dim)]
        self.newScheme = False
        '''

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
        '''
        if self.newScheme:
            #restore scheme to initial state
            initialPoints = np.linspace(self.a[0],self.b[0],2**1+1)
            self.refinement = [[(initialPoints[i],initialPoints[i+1],self.lmax[d]-1) for i in range(2**1)] for d in range(self.this_dim)]
            self.finestWidth = initialPoints[1] - initialPoints[0]
        else:
            #remove outdated refinement values
            for d in range(self.this_dim):
                for position in reversed(sorted(list(set(self.popArray[d])))):
                    self.refinement[d].pop(position)
        '''
        self.refinement.apply_remove(sort=True)
        self.refinement.refinement_postprocessing()
        self.refinement.reinit_new_objects()
        # self.combiintegral = 0.0

    '''
    def getErrors(self,integralarrayComplete, errorOperator, f):
        #errorArray = np.zeros(len(self.getAreas()))
        offsetAreas = np.ones(self.this_dim, dtype=int)
        for d in range(self.this_dim - 1):
            offsetAreas[d+1] = len(self.refinement[d]) * offsetAreas[d]
        for i in range(len(integralarrayComplete)):
            for j in range(len(integralarrayComplete[i][0])):
                position = self.getPosition(integralarrayComplete[i][0][j],self.this_dim)
                error = errorOperator.calcError(f,integralarrayComplete[i][0][j])
                for d in range(self.this_dim):
                    assert(len(position[d]) == 1)
                p = np.inner(np.ravel(position), offsetAreas)
                #errorArray[p] += error * integralarrayComplete[i][1]
        #self.errorArray = list(errorArray)
    '''

    def refine(self):
        # split all cells that have an error close to the max error
        areas = self.get_areas()
        self.prepare_refinement()
        self.refinement.clear_new_objects()
        margin = 0.1 #TODO not needed at the moment
        num_refinements = 0
        quit_refinement = False
        # self.refinement.print()
        while True:  # refine all areas for which area is within margin
            # get next area that should be refined
            if (self.useToleranceAsRefineCond == True):
                self.errorMax = self.tolerance
            found_object, position, refine_object = self.refinement.get_next_object_for_refinement(self.benefit_max * margin)
            # self.errorMax * margin
            if found_object and not quit_refinement:  # new area found for refinement
                self.refinements += 1
                num_refinements += 1
                # print("Refining position", position)
                quit_refinement = self.do_refinement(refine_object, position)

            else:  # all refinements done for this iteration -> reevaluate integral and check if further refinements necessary
                print("Finished refinement")
                print("Refined ", num_refinements, " times")
                self.refinement_postprocessing()
                break
        if self.refinements / 100 > self.counter:
            self.refinement.reinit_new_objects()
            self.combiintegral = 0
            self.subAreaIntegrals = []
            self.evaluationPerArea = []
            self.evaluationsTotal = 0
            self.counter += 1
            print("recalculating errors")

    def calc_terminate_condition_value(self):
        return abs(self.errorMax)
        # return abs(self.refinement.integral - self.realIntegral)