import math
import numpy as np
import abc, logging
from combiScheme import CombiScheme


# This class defines the general template of a Refinement Object that can be stored in the refinement container
class RefinementObject(object):
    def __init__(self, error_estimator):
        self.errorEstimator = error_estimator
        self.integral = None

    # set the local integral for area associated with RefinementObject
    def set_integral(self, integral):
        self.integral = integral

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


# This is the special class for the RefinementObject defined in the split extend scheme
class RefinementObjectExtendSplit(RefinementObject):
    def __init__(self, start, end, grid, number_of_refinements_before_extend,
                 parent_info,
                 coarseningValue=0,
                 needExtendScheme=0,
                 automatic_extend_split=False):
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
        self.integral = None
        # dictionary that maps all coarsened levelvectors to there uncoarsened ones
        # the can only be one uncoarsened levelvector for each coarsened one all other areas are set to 0
        self.levelvec_dict = {}
        self.grid = grid

        # initialize errors
        self.error = None

        self.automatic_extend_split = automatic_extend_split
        self.parent_info = parent_info if parent_info is not None else ParentInfo()
        self.switch_to_parent_estimation = self.grid.is_high_order_grid()
        self.children = []

    # this routine decides if we split or extend the RefinementObject
    def refine(self):
        correction = 0.0
        coarsening_level = self.coarseningValue
        benefit_extend = benefit_split = None
        if self.automatic_extend_split:
            if self.switch_to_parent_estimation:
                comparison = self.sum_siblings
                num_comparison = self.evaluations * 2 ** self.dim
            else:
                comparison = self.integral
                num_comparison = self.evaluations

            benefit_split = self.parent_info.get_split_benefit(integral=self.integral, num_comparison=num_comparison,
                                                               grid=self.grid)
            benefit_extend = self.parent_info.get_extend_benefit(comparison=comparison, num_comparison=num_comparison,
                                                                 grid=self.grid)

            if self.switch_to_parent_estimation:
                benefit_extend += self.parent_info.get_extend_error_correction()

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
            benefit_extend = benefit_extend - correction if benefit_extend is not None and self.grid.is_high_order_grid() else None
            num_points_split_parent = self.parent_info.num_points_split_parent if self.grid.is_high_order_grid else None
            parent_info = ParentInfo(previous_integral=self.integral, parent=self.parent_info.parent,
                                     num_points_extend_parent=self.evaluations,
                                     benefit_extend=benefit_extend, level_parent=self.parent_info.level_parent,
                                     num_points_split_parent=num_points_split_parent)
            newRefinementObject = RefinementObjectExtendSplit(start=self.start, end=self.end, grid=self.grid,
                                                              number_of_refinements_before_extend=self.numberOfRefinementsBeforeExtend,
                                                              parent_info=parent_info,
                                                              coarseningValue=coarseningValue,
                                                              needExtendScheme=self.needExtendScheme,
                                                              automatic_extend_split=self.automatic_extend_split)
            self.children.append(newRefinementObject)
            return [newRefinementObject], lmaxIncrease, update_other_coarsenings

        elif (self.automatic_extend_split and benefit_extend >= benefit_split) or (
                not self.automatic_extend_split and self.needExtendScheme >= 0):  # split the array
            # add to integralArray
            self.needExtendScheme += 1
            # print("Splitting", self.start, self.end)
            newRefinementObjects = self.split_area_arbitrary_dim()
            return newRefinementObjects, None, None
        else:
            print("Error!!!! Invalid value")
            assert False

    # in case lmax was changed the coarsening value of other RefinementObjects need to be increased
    def update(self, update_info):
        self.coarseningValue += update_info
        self.levelvec_dict = {}

    def add_level(self, levelvec_coarsened, levelvec):
        # print("adding", levelvec_coarsened, levelvec)
        self.levelvec_dict[levelvec_coarsened] = levelvec

    def is_already_calculated(self, levelvec_coarsened, levelvec):
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
            parent_info = ParentInfo(parent=self, last_refinement_split=True)
            new_refinement_object = RefinementObjectExtendSplit(start=start_sub_area, end=end_sub_area, grid=self.grid,
                                                                number_of_refinements_before_extend=self.numberOfRefinementsBeforeExtend,
                                                                parent_info=parent_info,
                                                                coarseningValue=self.coarseningValue,
                                                                needExtendScheme=self.needExtendScheme,
                                                                automatic_extend_split=self.automatic_extend_split)
            self.children.append(new_refinement_object)
            sub_area_array.append(new_refinement_object)
        return sub_area_array

    # set the local error associated with RefinementObject
    def set_error(self, error):
        if self.switch_to_parent_estimation and self.parent_info.last_refinement_split:
            error /= 2 ** self.dim
            # print("Reduced error")
        self.error = error


# This class is used in the Extend Split RefinementObject as a container
# to store various variables used for the error estimates
class ParentInfo(object):
    def __init__(self, previous_integral=None, parent=None, split_parent_integral=None, extend_parent_integral=None,
                 level_parent=-1,
                 num_points_split_parent=None, num_points_extend_parent=None, benefit_extend=None, benefit_Split=None,
                 sum_siblings=None, last_refinement_split=False):
        self.previous_integral = previous_integral
        self.parent = parent
        self.split_parent_integral = split_parent_integral
        self.extend_parent_integral = extend_parent_integral
        self.level_parent = level_parent
        self.num_points_split_parent = num_points_split_parent
        self.num_points_extend_parent = num_points_extend_parent
        self.benefit_extend = benefit_extend
        self.benefit_Split = benefit_Split
        self.num_points_reference = None
        self.sum_siblings = sum_siblings
        self.comparison = None
        self.extend_error_correction = 0
        self.last_refinement_split = last_refinement_split

    def get_extend_benefit(self, comparison, num_comparison, grid):
        if self.benefit_extend is not None:
            return self.benefit_extend
        assert num_comparison > self.num_points_extend_parent
        error_extend = abs((self.split_parent_integral - comparison) / (abs(comparison) + 10 ** -100))
        if not grid.is_high_order_grid():
            return error_extend * (self.num_points_split_parent - self.num_points_reference)
        else:
            return error_extend * (self.num_points_split_parent)

    def get_split_benefit(self, integral, num_comparison, grid):
        if self.benefit_Split is not None:
            return self.benefit_Split
        assert num_comparison > self.num_points_split_parent or self.switch_to_parent_estimation
        if grid.boundary:
            assert self.num_points_split_parent > 0
        error_split = abs((self.extend_parent_integral - integral) / (abs(integral) + 10 ** -100))
        if not grid.is_high_order_grid():
            return error_split * (self.num_points_extend_parent - self.num_points_reference)
        else:
            return error_split * (self.num_points_extend_parent)

    def get_extend_error_correction(self):
        return self.extend_error_correction * self.num_points_split_parent


# This is the special class for the RefinementObject defined in the split extend scheme
class RefinementObjectCell(RefinementObject):
    cell_dict = {}
    cells_for_level = []
    punish_depth = False

    def __init__(self, start, end, levelvec, a, b, lmin, father=None):
        self.a = a
        self.b = b
        self.lmin = lmin
        # start of subarea
        self.start = start
        # end of subarea
        self.end = end
        RefinementObjectCell.cell_dict[self.get_key()] = self
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
        self.integral = None
        for d in range(self.dim):
            parent = RefinementObjectCell.parent_cell_arbitrary_dim(d, self.levelvec, self.start, self.end, self.a,
                                                                    self.b, self.lmin)
            if parent is not None:
                self.parents.append(parent)
                if parent != father:
                    # print(parent, RefinementObjectCell.cell_dict.items(), self.get_key(), father, self.levelvec)
                    parent_object = RefinementObjectCell.cell_dict[parent]
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
                if candidate in RefinementObjectCell.cell_dict:
                    continue
                # print("candidate", candidate)
                # key = candidate.get_key()
                can_be_refined = True
                for parent in RefinementObjectCell.get_parents(levelvec_copy, candidate[0], candidate[1], self.a,
                                                               self.b, self.dim, self.lmin):
                    if parent not in RefinementObjectCell.cell_dict or RefinementObjectCell.cell_dict[
                            parent].isActive():
                        can_be_refined = False
                        break
                if can_be_refined:
                    new_objects.append(
                        RefinementObjectCell(candidate[0], candidate[1], list(levelvec_copy), self.a, self.b, self.lmin,
                                             self.get_key()))

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
            if child not in RefinementObjectCell.cell_dict:
                new_refinement_object = RefinementObjectCell(child[0], child[1], list(levelvec), self.a, self.b,
                                                             self.lmin, self.get_key())
                # RefinementObjectCell.cells_for_level[self.level+1].append(new_refinement_object)
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
            if self.punish_depth:
                self.error *= np.prod(np.array(self.end) - np.array(self.start))
        else:
            self.error = 0
        # print("Error of refine object:", self.get_key(), "is:", self.error)

        self.sub_integrals = []


# This is the special class for the RefinementObject defined in the split extend scheme
class RefinementObjectSingleDimension(RefinementObject):
    def __init__(self, start, end, dim, coarsening_level=0):
        # start of subarea
        self.start = start
        # end of subarea
        self.end = end
        self.dim = dim
        # indicates how often area needs to be coarsened according to extend scheme
        self.coarsening_level = coarsening_level
        self.evaluations = 0

    def refine(self):
        # coarseningLevel = self.refinement[dimValue][area[dimValue]][2]
        # if coarseningLevel == 0 and allDimsMaxRefined == False: #not used currently
        #    continue
        # positionDim = area[dimValue]
        # if positionDim in self.popArray[dimValue]: #do not refine one interval twice
        #    continue
        # print("refine:", positionDim , "in dim:", dimValue, "coarsening Level:", coarseningLevel)
        # splitAreaInfo = self.refinement[dimValue][positionDim]
        # lowerBorder = splitAreaInfo[0]
        # upperBorder = splitAreaInfo[1]
        # self.popArray[dimValue].append(positionDim)
        lmax_increase = None
        update = None
        coarsening_value = 0
        # if we are already at maximum refinement coarsening_value stays at 0
        if self.coarsening_level == 0:
            coarsening_value = 0
        else:  # otherwise decrease coarsening level
            coarsening_value = self.coarsening_level - 1
        # in case we have refined complete scheme (i.e. coarensingLevel was 0) we have to increase level everywhere else
        if (self.coarsening_level == 0):  # extend scheme if we are at maximum refinement
            # increase coarsening level in all other dimensions
            for d in range(self.dim):
                for i in range(len(self.refinement[d])):
                    icoarsen = self.refinement[d][i][2] + 1
                    self.refinement[d][i] = (self.refinement[d][i][0], self.refinement[d][i][1], icoarsen)
            # increase lmax by 1
            lmax_increase = [1 for d in range(self.dim)]
            update = 1
            # print("New scheme")
            # self.scheme = getCombiScheme(self.lmin[0],self.lmax[0],self.dim)
            # self.newScheme = True
        # add new refined interval to refinement array (it has half of the width)
        new_width = (self.end - self.start) / 2.0
        new_objects = []
        new_objects.append(
            RefinementObjectSingleDimension(self.start, self.start + new_width, self.dim, coarsening_value))
        new_objects.append(
            RefinementObjectSingleDimension(self.start + new_width, self.end, self.dim, coarsening_value))
        # self.finestWidth = min(new_width,self.finestWidth)
        return new_objects, lmax_increase, update

    # in case lmax was changed the coarsening value of other RefinementObjects need to be increased
    def update(self, update_info):
        self.coarsening_level += update_info
