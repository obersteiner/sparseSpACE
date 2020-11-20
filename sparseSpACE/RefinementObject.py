import math
import numpy as np
import abc, logging
from sparseSpACE.Grid import *

# This class defines the general template of a Refinement Object that can be stored in the refinement container
class RefinementObject(object):
    def __init__(self, error_estimator):
        self.errorEstimator = error_estimator
        self.value = None

    # set the local value for area associated with RefinementObject
    def set_value(self, value):
        self.value = value

    # set the evaluations that were performed in the refinementobject
    def set_evaluations(self, evaluations):
        self.evaluations = evaluations

    # refine this object and return newly created objects
    @abc.abstractmethod
    def refine(self):
        pass

    # set the local error associated with RefinementObject
    def set_error(self, error):
        self.error = error

    def reinit(self):
        pass

    def add_evaluations(self, evaluations):
        self.evaluations += evaluations

    def point_in_area(self, point):
        for d in range(self.dim):
            if point[d] < self.start[d] or point[d] > self.end[d]:
                return False
        return True

# This class is used in the Extend Split RefinementObject as a container
# to store various variables used for the error estimates
class ErrorInfo(object):
    def __init__(self, previous_value=None, parent=None, split_parent_integral=None, extend_parent_integral=None,
                 level_parent=-1,
                 num_points_split_parent=None, num_points_extend_parent=None, benefit_extend=None, benefit_Split=None,
                 sum_siblings=None, last_refinement_split=False):
        self.previous_value = previous_value
        self.parent = parent
        self.split_parent_integral = split_parent_integral
        self.extend_parent_integral = extend_parent_integral
        self.level_parent = level_parent
        self.num_points_split_parent = num_points_split_parent
        self.num_points_extend_parent = num_points_extend_parent
        self.benefit_extend = benefit_extend
        self.benefit_split = benefit_Split
        self.num_points_reference = None
        self.sum_siblings = sum_siblings
        self.comparison = None
        self.extend_error_correction = None
        self.last_refinement_split = last_refinement_split


    def get_extend_benefit(self):
        return self.benefit_extend

    def get_split_benefit(self):
        return self.benefit_split

    def get_extend_error_correction(self):
        return self.extend_error_correction


# This is the special class for the RefinementObject defined in the split extend scheme
class RefinementObjectExtendSplit(RefinementObject):
    def __init__(self, start: Sequence[float], end: Sequence[float], grid: Grid, number_of_refinements_before_extend: int =2,
                 parent_info: ErrorInfo =None,
                 coarseningValue: int=0,
                 needExtendScheme: int=0,
                 automatic_extend_split: bool=False,
                 splitSingleDim: bool=True):
        # start of subarea
        self.start = start
        # end of subarea
        self.end = end
        self.dim = len(start)
        # indicates how often area needs to be coarsened according to extend scheme
        self.coarseningValue = coarseningValue
        # indicates how many splits where already performed for this region
        self.needExtendScheme = needExtendScheme
        # indicates after how many splits only extends are performed
        self.numberOfRefinementsBeforeExtend = number_of_refinements_before_extend
        self.evaluations = 0
        self.value = None
        # dictionary that maps all coarsened levelvectors to there uncoarsened ones
        # the can only be one uncoarsened levelvector for each coarsened one all other areas are set to 0
        self.levelvec_dict = {}
        self.grid = grid
        self.twins = [None] * self.dim

        # initialize errors
        self.error = None
        self.twinErrors = [None] * self.dim

        self.splitSingleDim = splitSingleDim
        self.automatic_extend_split = automatic_extend_split
        self.parent_info = parent_info if parent_info is not None else ErrorInfo()
        self.switch_to_parent_estimation = self.grid.is_high_order_grid()
        self.children = []

    # this routine decides if we split or extend the RefinementObject
    def refine(self) -> Tuple[RefinementObject, Sequence[int], int]:
        correction = 0.0
        coarsening_level = self.coarseningValue
        benefit_extend = benefit_split = None
        if self.automatic_extend_split:


            benefit_split = self.parent_info.get_split_benefit()
            benefit_extend = self.parent_info.get_extend_benefit()
            #print("Benefit split", benefit_split, "Benefit extend", benefit_extend)
            if self.switch_to_parent_estimation:
                correction = abs(self.parent_info.get_extend_error_correction())
                benefit_extend += correction

        if (self.automatic_extend_split and benefit_extend < benefit_split) or (
                not self.automatic_extend_split and
                self.needExtendScheme >= self.numberOfRefinementsBeforeExtend):  # add new component grids to scheme and refine only target area

            if self.coarseningValue == 0:
                coarseningValue = 0
            else:
                coarseningValue = coarsening_level - 1
            # in case we have refined complete scheme (i.e. coarensingLevel was 0)
            # we have to increase level everywhere else

            if coarsening_level == 0:
                # increase lmax by dim
                lmaxIncrease = [1 for d in range(self.dim)]
                update_other_coarsenings = 1

            else:
                lmaxIncrease = None
                update_other_coarsenings = None
            #print("Performing extend for", self.start, self.end, benefit_extend, benefit_split, self.parent_info.num_points_split_parent, self.parent_info.num_points_extend_parent, self.parent_info.split_parent_integral, self.parent_info.extend_parent_integral, self.integral, self.sum_siblings)
            benefit_extend = benefit_extend - correction if benefit_extend is not None and self.grid.is_high_order_grid() else None
            num_points_split_parent = self.parent_info.num_points_split_parent if self.grid.is_high_order_grid() else None
            parent_info = ErrorInfo(previous_value=self.value if self.value is not None else self.value, parent=self.parent_info.parent,
                                     num_points_extend_parent=self.evaluations,
                                     benefit_extend=benefit_extend, level_parent=self.parent_info.level_parent,
                                     num_points_split_parent=num_points_split_parent)
            newRefinementObject = RefinementObjectExtendSplit(start=self.start, end=self.end, grid=self.grid,
                                                              number_of_refinements_before_extend=self.numberOfRefinementsBeforeExtend,
                                                              parent_info=parent_info,
                                                              coarseningValue=coarseningValue,
                                                              needExtendScheme=self.needExtendScheme,
                                                              automatic_extend_split=self.automatic_extend_split,
                                                              splitSingleDim=self.splitSingleDim)
            newRefinementObject.twins = self.twins
            newRefinementObject.twinErrors = self.twinErrors
            self.children.append(newRefinementObject)
            return [newRefinementObject], lmaxIncrease, update_other_coarsenings

        elif (self.automatic_extend_split and benefit_extend >= benefit_split) or (
                not self.automatic_extend_split and self.needExtendScheme >= 0):  # split the array
            # add to integralArray
            # print("Splitting", self.start, self.end)
            if self.splitSingleDim:
                '''
                d = self.get_split_dim()
                print("Split in dimension", d, ", maxTwinError =", self.twinErrors[d]) #TODO
                newRefinementObjects = self.split_area_single_dim(d)
                return newRefinementObjects, None, None
                '''
                dims = self.get_split_dims()
                #dims = [self.get_split_dim()]
                newRefinementObjects = [self]
                for d in dims:
                    newObjects = []
                    print("Split in dimension", d, ", maxTwinError =", self.twinErrors[d], self.twinErrors) #TODO
                    for area in newRefinementObjects:
                        newObjects.extend(area.split_area_single_dim(d))
                    newRefinementObjects = newObjects
                assert len(newRefinementObjects) == 2**len(dims)
                for area in newRefinementObjects:
                    for d in dims[:]:
                        area.twins[d] = None
                for i in range(len(newRefinementObjects)):
                    area = newRefinementObjects[i]
                    for d in range(len(dims)):
                        if area.twins[dims[d]] is None:
                            twin_distance = 2**(len(dims) - d - 1)
                            twin = newRefinementObjects[i + twin_distance]
                            assert twin is not None
                            area.set_twin(dims[d], twin)
                            #print("Area", area.start, area.end, "has twin", twin.start, twin.end, "in dimension", dims[d])
                    for d in range(self.dim):
                        assert area.twins[d] is not None
                        
                return newRefinementObjects, None, None

            else:
                newRefinementObjects = self.split_area_arbitrary_dim()
                return newRefinementObjects, None, None
        else:
            print("Error!!!! Invalid value")
            assert False

    # in case lmax was changed the coarsening value of other RefinementObjects need to be increased
    def update(self, update_info: int):
        self.coarseningValue += update_info
        self.levelvec_dict = {}

    def add_level(self, levelvec_coarsened: Tuple[int, ...], levelvec: Sequence[int]):
        # print("adding", levelvec_coarsened, levelvec)
        self.levelvec_dict[levelvec_coarsened] = levelvec

    # this method indicates whether for this area already a grid has been calculated with by another levelvector with the same
    # levelvector_coarsened. This is necessary since collisions in coarsening process are guaranteed to happen.
    # The levelvec parameter is the uncoarsened parameter.
    def is_already_calculated(self, levelvec_coarsened: Tuple[int, ...], levelvec: Tuple[int, ...]):
        if levelvec_coarsened not in self.levelvec_dict:
            return False
        else:
            # print(levelvec_coarsened, levelvec, self.levelvec_dict[levelvec_coarsened])
            return self.levelvec_dict[levelvec_coarsened] != levelvec

    # splits the current area into 2**dim smaller ones and returns them
    def split_area_arbitrary_dim(self):
        dim = self.dim
        num_sub_areas = 2 ** dim
        start = self.start
        end = self.end
        midpoint = [self.grid.get_mid_point(start[d], end[d], d) for d in range(self.dim)]
        sub_area_array = []
        for i in range(num_sub_areas):
            start_sub_area = np.zeros(dim)
            end_sub_area = np.zeros(dim)
            rest = i
            for d in reversed(list(range(dim))):
                start_sub_area[d] = start[d] if rest < 2 ** d else midpoint[d]
                end_sub_area[d] = midpoint[d] if rest < 2 ** d else end[d]
                rest = rest % 2 ** d
            parent_info = ErrorInfo(parent=self, last_refinement_split=True)
            new_refinement_object = RefinementObjectExtendSplit(start=start_sub_area, end=end_sub_area, grid=self.grid,
                                                                number_of_refinements_before_extend=self.numberOfRefinementsBeforeExtend,
                                                                parent_info=parent_info,
                                                                coarseningValue=self.coarseningValue,
                                                                needExtendScheme=self.needExtendScheme + 1,
                                                                automatic_extend_split=self.automatic_extend_split,
                                                                splitSingleDim=self.splitSingleDim)
            self.children.append(new_refinement_object)
            sub_area_array.append(new_refinement_object)
        return sub_area_array
    
    # splits the current area in the d-th dimension and returns the pair of twins
    def split_area_single_dim(self, d):
        midpoint = self.grid.get_mid_point(self.start[d], self.end[d], d)
        sub_area_array = []
        for i in range(2):
            start_sub_area = list(self.start)
            end_sub_area = list(self.end)
            start_sub_area[d] = start_sub_area[d] if i == 0 else midpoint
            end_sub_area[d] = midpoint if i == 0 else end_sub_area[d]
            parent_info = ErrorInfo(parent=self, last_refinement_split=True)
            new_refinement_object = RefinementObjectExtendSplit(start=start_sub_area, end=end_sub_area, grid=self.grid,
                                                                number_of_refinements_before_extend=self.numberOfRefinementsBeforeExtend,
                                                                parent_info=parent_info,
                                                                coarseningValue=self.coarseningValue,
                                                                needExtendScheme=self.needExtendScheme + 1,
                                                                automatic_extend_split=self.automatic_extend_split,
                                                                splitSingleDim=self.splitSingleDim)
            new_refinement_object.twins = list(self.twins)
            new_refinement_object.twinErrors = list([t * 0.5 if t is not None else t for t in self.twinErrors])
            new_refinement_object.twinErrors[d] = None
            self.children.append(new_refinement_object)
            sub_area_array.append(new_refinement_object)
        sub_area_array[0].set_twin(d, sub_area_array[1])       
        return sub_area_array

    # set the local error associated with RefinementObject
    def set_error(self, error):
        if self.switch_to_parent_estimation and self.parent_info.last_refinement_split:
            error /= 2 ** self.dim
            # print("Reduced error")
        self.error = error

    # define two area objects to be twins of each other in the d-th dimension
    def set_twin(self, d, twin):
        self.twins[d] = twin
        twin.twins[d] = self

    # set the twin error in dimension d of the given area object to the given value and update the dimension in which the twin error is maximum
    def set_twin_error(self, d, twinError):
        twinError += 10**-14
        twin = self.twins[d]
        twin.twinErrors[d] = self.twinErrors[d] = twinError

    # returns the dimension in which the split shall be performed
    def get_split_dim(self):
        return np.argmax(self.twinErrors)

    def get_split_dims(self, threshold=0.9):
        print("Twin errors for", self.start, self.end, "are", self.twinErrors, "twins are", self.twins[0].start, self.twins[0].end, self.twins[1].start, self.twins[1].end,)
        assert 0 <= threshold <= 1
        max_error = max(self.twinErrors)        
        dims_for_refinement = []        
        for d in range(self.dim):
            if self.twinErrors[d] >= max_error * threshold:
                dims_for_refinement.append(d)
        return dims_for_refinement

    def contains(self, point):
        contained = True
        for d in range(self.dim):
            if point[d] < self.start[d] or point[d] > self.end[d]:
                contained = False
                break
        return contained

    def subset_of_contained_points(self, points):
        contained_points = []
        for p in points:
            if self.contains(p):
                contained_points.append(p)
        return contained_points


# This is the special class for the RefinementObject defined in the split extend scheme
class RefinementObjectCell(RefinementObject):
    #punish_depth = False

    def __init__(self, start, end, levelvec, a, b, lmin, cell_dict, father=None):
        self.a = a
        self.b = b
        self.lmin = lmin
        # start of subarea
        self.start = start
        # end of subarea
        self.end = end
        self.cell_dict = cell_dict
        self.cell_dict[self.get_key()] = self
        self.dim = len(start)
        self.levelvec = np.array(levelvec, dtype=int)
        # print("levelvec", self.levelvec)
        self.level = sum(levelvec) - self.dim + 1
        # self.father = father
        self.children = []
        # self.descendants = set()
        self.error = None
        self.active = True
        self.parents = []
        self.sub_integrals = []
        self.value = None
        for d in range(self.dim):
            parent = RefinementObjectCell.parent_cell_arbitrary_dim(d, self.levelvec, self.start, self.end, self.a,
                                                                    self.b, self.lmin)
            if parent is not None:
                self.parents.append(parent)
                if parent != father:
                    # print(parent, RefinementObjectCell.cell_dict.items(), self.get_key(), father, self.levelvec)
                    parent_object = self.cell_dict[parent]
                    parent_object.add_child(self)

    def add_child(self, child):
        self.children.append(child)

    def get_key(self):
        return tuple((tuple(self.start), tuple(self.end)))

    def isActive(self):
        return self.active

    def contains(self, point):
        contained = True
        for d in range(self.dim):
            if point[d] < self.start[d] or point[d] > self.end[d]:
                contained = False
                break
        return contained

    def subset_of_contained_points(self, points):
        contained_points = []
        for p in points:
            if self.contains(p):
                contained_points.append(p)
        return contained_points

    def is_corner(self, point):
        is_corner = True
        for d in range(self.dim):
            if point[d] != self.start[d] and point[d] != self.end[d]:
                is_corner = False
                break
        return is_corner

    def refine(self):
        assert self.active
        self.active = False
        new_objects = []
        for d in range(self.dim):
            levelvec_copy = list(self.levelvec)
            levelvec_copy[d] += 1
            possible_candidates_d = RefinementObjectCell.children_cell_arbitrary_dim(d, self.start, self.end, self.dim)
            for candidate in possible_candidates_d:
                if candidate in self.cell_dict:
                    continue
                # print("candidate", candidate)
                # key = candidate.get_key()
                can_be_refined = True
                for parent in RefinementObjectCell.get_parents(levelvec_copy, candidate[0], candidate[1], self.a,
                                                               self.b, self.dim, self.lmin):
                    if parent not in self.cell_dict or self.cell_dict[
                            parent].isActive():
                        can_be_refined = False
                        break
                if can_be_refined:
                    new_objects.append(
                        RefinementObjectCell(candidate[0], candidate[1], list(levelvec_copy), self.a, self.b, self.lmin, cell_dict=self.cell_dict,
                                             father=self.get_key()))

        self.children.extend(new_objects)
        # print("New refined objects", [object.get_key() for object in new_objects])
        return new_objects, None, None

    # splits the current cell into the 2 children in the dimension d and returns the children
    def split_cell_arbitrary_dim(self, d):
        childs = RefinementObjectCell.children_cell_arbitrary_dim(d, self.start, self.end, self.dim)
        sub_area_array = []
        levelvec = list(self.levelvec)
        levelvec[d] += 1
        for child in childs:
            if child not in self.cell_dict:
                new_refinement_object = RefinementObjectCell(child[0], child[1], list(levelvec), self.a, self.b,
                                                             self.lmin, cell_dict=self.cell_dict, father=self.get_key())
                sub_area_array.append(new_refinement_object)
        return sub_area_array

    # splits the current cell into the 2 children in the dimension d and returns the children
    @staticmethod
    def children_cell_arbitrary_dim(d, start, end, dim):
        spacing = np.zeros(dim)
        spacing[d] = 0.5 * (end[d] - start[d])
        start_sub_area = np.array(start)
        end_sub_area = np.array(end)
        child1 = tuple((tuple(start_sub_area + spacing), tuple(end_sub_area)))
        child2 = tuple((tuple(start_sub_area), tuple(end_sub_area - spacing)))
        return [child1, child2]

    @staticmethod
    def get_parents(levelvec, start, end, a, b, dim, lmin):
        parents = []
        for d in range(dim):
            parent = RefinementObjectCell.parent_cell_arbitrary_dim(d, levelvec, start, end, a, b, lmin)
            if parent is not None:
                parents.append(parent)
        return parents

    @staticmethod
    def parent_cell_arbitrary_dim(d, levelvec, start, end, a, b, lmin):
        levelvec_parent = list(levelvec)
        if levelvec_parent[d] <= lmin[d]:
            return None
        levelvec_parent[d] = levelvec_parent[d] - 1
        parent_start = np.array(start)
        parent_end = np.array(end)
        index_of_start = start[d] * 2 ** levelvec[d]
        if index_of_start % 2 == 1:  # start needs to be changed
            parent_start[d] = parent_end[d] - (b[d] - a[d]) / 2 ** levelvec_parent[d]
        else:  # end needs to be changed
            parent_end[d] = parent_start[d] + (b[d] - a[d]) / 2 ** levelvec_parent[d]

        return tuple((tuple(parent_start), tuple(parent_end)))

    def get_points(self):
        return set(zip(*[g.ravel() for g in np.meshgrid(*[[self.start[d], self.end[d]] for d in range(self.dim)])]))

    # sets the error of a cell only if it is refinable; otherwise we do not need to set it here
    def set_error(self, error):
        # only define an error if active cell
        if self.active:
            self.error = error
            #if self.punish_depth:
            #    self.error *= np.prod(np.array(self.end) - np.array(self.start))
        else:
            self.error = 0
        # print("Error of refine object:", self.get_key(), "is:", self.error)

        self.sub_integrals = []


from scipy import optimize

# This is the special class for the RefinementObject defined in the single dimension refinement scheme
class RefinementObjectSingleDimension(RefinementObject):
    def __init__(self, start, end, this_dim, dim, levels, grid, a, b, chebyshev=False, coarsening_level=0):
        # start of subarea
        self.start = start
        # end of subarea
        self.end = end
        self.this_dim = this_dim
        self.dim = dim
        # indicates how often area needs to be coarsened according to extend scheme
        self.coarsening_level = coarsening_level
        # number of evaluations
        self.evaluations = 0
        # integral in this area
        self.value = 0.0
        # volume of this area
        self.volume = None
        # level at start and end as point levels (as tuple)
        self.levels = levels
        self.error = 0.0
        self.benefit = None
        self.a = a
        self.b = b
        self.chebyshev = chebyshev
        # The middle between two nodes can be calculated with the probability if
        # available so that infinite boundaries are possible
        self.grid = grid

        assert(end > start)

    def refine(self):
        # coarseningLevel = self.refinement[dimValue][area[dimValue]][2]
        # if coarseningLevel == 0 and allDimsMaxRefined == False: #not used currently
        #    continue
        # positionDim = area[dimValue]
        # if positionDim in self.popArray[dimValue]: #do not refine one interval twice
        #    continue
        # print("refine:", positionDim , "in this_dim:", dimValue, "coarsening Level:", coarseningLevel)
        # splitAreaInfo = self.refinement[dimValue][positionDim]
        # lowerBorder = splitAreaInfo[0]
        # upperBorder = splitAreaInfo[1]
        # self.popArray[dimValue].append(positionDim)

        lmax_increase = None
        update = None
        #coarsening_value = 0
        # if we are already at maximum refinement coarsening_value stays at 0
        if self.coarsening_level == 0:
            coarsening_value = 0
        else:  # otherwise decrease coarsening level
            coarsening_value = self.coarsening_level - 1
        # in case we have refined complete scheme (i.e. coarensingLevel was 0) we have to increase level everywhere else
        #if (self.coarsening_level == 0):  # extend scheme if we are at maximum refinement
        #    # increase lmax by 1
        #    lmax_increase = [1 if d == self.this_dim or self.dim_adaptive == False else 0 for d in range(self.dim)]
        #    update = 1
        #    # print("New scheme")
        #    # self.scheme = getCombiScheme(self.lmin[0],self.lmax[0],self.this_dim)
        #    # self.newScheme = True
        # add new refined interval to refinement array
        if not self.chebyshev:
            mid = self.grid.get_mid_point(self.start, self.end, self.this_dim)
        else:
            mid = self.map_chebyshev(self.start, self.end)
        assert self.start < mid < self.end, "{} < {} < {} does not hold.".format(self.start, mid, self.end)

        newObjects = []
        newLevel = max(self.levels) + 1
        # print("newLevel", newLevel)
        newObjects.append(RefinementObjectSingleDimension(self.start, mid, self.this_dim, self.dim, list((self.levels[0], newLevel)), grid=self.grid, coarsening_level=coarsening_value, a=self.a, b=self.b, chebyshev=self.chebyshev))
        newObjects.append(RefinementObjectSingleDimension(mid, self.end, self.this_dim, self.dim, list((newLevel, self.levels[1])), grid=self.grid, coarsening_level=coarsening_value, a=self.a, b=self.b, chebyshev=self.chebyshev))
        # self.finestWidth = min(newWidth,self.finestWidth)
        return newObjects, lmax_increase, update

    # in case lmax was changed the coarsening value of other RefinementObjects need to be increased
    def update(self, update_info):
        self.coarsening_level += update_info
        assert self.coarsening_level >= 0

    def print(self):
        print("refineObjSingleDim: ", self.start, "\t--\t", self.end, " \tthis_dim:", self.this_dim, "\terror:", self.error, "\tlevels:", self.levels, "\tvolume:", self.volume, "benefit:", self.benefit, "\tcoarsening", self.coarsening_level)

    def set_levels(self, levels):
        self.levels = levels

    def get_width(self):
        return self.end - self.start

    def set_volume(self, volume):
        self.volume = volume
        # self.error = abs(volume)

    def add_volume(self, volume):
        assert isinstance(volume, np.ndarray)
        if self.volume is None:
            self.volume = volume
        else:
            self.volume += volume

    def reinit(self):
        self.volume = None
        self.error = 0.0
        self.value = 0.0
        self.evaluations = 0

    def map_chebyshev(self, start, end) -> float:
        coordinate_start = math.acos(1 - 2*(start - self.a) / (self.b - self.a)) / math.pi
        coordinate_end = math.acos(1 - 2*(end - self.a) / (self.b - self.a)) / math.pi
        coordinate = (coordinate_start + coordinate_end) / 2
        print("Start", coordinate)
        coordinate_normalized = (coordinate - self.a)/(self.b - self.a)
        coordinate = self.a + (self.b- self.a) * (1 - math.cos(coordinate_normalized * math.pi)) / 2
        #coordinate = (coordinate + 1 + self.a[d]) / (2 * (self.b[d] - self.a[d]))
        print("Transformed", coordinate, coordinate_normalized)
        return coordinate
