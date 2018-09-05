
import math
import numpy as np
import abc,logging


# This class defines the general template of a Refinement Object that can be stored in the refinement container
class RefinementObject(object):
    def __init__(self, errorEstimator):
        self.errorEstimator = errorEstimator
        self.integral = None

    # set the local integral for area associated with RefinementObject
    def set_integral(self, integral):
        self.integral = integral

    # refine this object and return newly created objects
    @abc.abstractmethod
    def refine(self):
        pass

    # set the local error associated with RefinementObject
    def set_error(self, error):
        self.error = error


# This is the special class for the RefinementObject defined in the split extend scheme
class RefinementObjectExtendSplit(RefinementObject):
    def __init__(self, start, end, number_of_refinements_before_extend, coarseningValue=0, needExtendScheme=0):
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

    # this routine decides if we split or extend the RefinementObject
    def refine(self):
        coarsening_level = self.coarseningValue
        if (
                self.needExtendScheme >= self.numberOfRefinementsBeforeExtend):  # add new component grids to scheme and refine only target area

            if self.coarseningValue == 0:
                self.coarseningValue = 0
            else:
                self.coarseningValue = coarsening_level - 1
            # in case we have refined complete scheme (i.e. coarensingLevel was 0)
            # we have to increase level everywhere else
            if coarsening_level == 0:
                # increase lmax by dim
                lmaxIncrease = [1 for d in range(self.dim)]
                newRefinementObject = RefinementObjectExtendSplit(self.start, self.end,
                                                                  self.numberOfRefinementsBeforeExtend,
                                                                  self.coarseningValue, self.needExtendScheme)
                # self.refinement.append((lowerBorder, upperBorder , factor, coarseningValue, needExtendScheme))
                # self.newRefinementArray.append((lowerBorder, upperBorder , factor, coarseningValue, needExtendScheme))
                # self.errorArray = list(self.errorArrayWithoutCost)
                # self.combiintegral = 0.0
                # self.subAreaIntegrals = []
                # self.evaluationPerArea = []
                # self.evaluationsTotal = 0
                # self.checkCombiScheme()
                return [newRefinementObject], lmaxIncrease, 1
            else:
                # add to integralArray
                newRefinementObject = RefinementObjectExtendSplit(self.start, self.end,
                                                                  self.numberOfRefinementsBeforeExtend,
                                                                  self.coarseningValue, self.needExtendScheme)
                return [newRefinementObject], None, None
        elif self.needExtendScheme >= 0:  # split the array
            # add to integralArray
            self.needExtendScheme += 1
            newRefinementObjects = self.split_area_arbitrary_dim()
            return newRefinementObjects, None, None
        else:
            print("Error!!!! Invalid value")
            assert False

    # in case lmax was changed the coarsening value of other RefinementObjects need to be increased
    def update(self, update_info):
        self.coarseningValue += update_info

    # splits the current area into 2**dim smaller ones and returns them
    def split_area_arbitrary_dim(self):
        dim = self.dim
        num_sub_areas = 2 ** dim
        start = self.start
        spacing = 0.5 * (self.end - start)
        sub_area_array = []
        for i in range(num_sub_areas):
            start_sub_area = np.zeros(dim)
            end_sub_area = np.zeros(dim)
            rest = i
            for d in reversed(list(range(dim))):
                start_sub_area[d] = start[d] + int(rest / 2 ** d) * spacing[d]
                end_sub_area[d] = start[d] + (int(rest / 2 ** d) + 1) * spacing[d]
                rest = rest % 2 ** d
            new_refinement_object = RefinementObjectExtendSplit(start_sub_area, end_sub_area,
                                                                self.numberOfRefinementsBeforeExtend,
                                                                self.coarseningValue, self.needExtendScheme)
            sub_area_array.append(new_refinement_object)
        return sub_area_array


# This is the special class for the RefinementObject defined in the split extend scheme
class RefinementObjectCell(RefinementObject):
    def __init__(self, start, end, coefficient):
        # start of subarea
        self.start = start
        # end of subarea
        self.end = end
        self.dim = len(start)
        self.coefficient = coefficient

    # this routine decides if we split or extend the RefinementObject
    def refine(self):
        return self.split_cell_arbitrary_dim(), None, None
        self.coefficient = - (self.dim - 1)

    # splits the current area into 2**dim smaller ones and returns them
    def split_cell_arbitrary_dim(self):
        dim = self.dim
        num_sub_areas = 2 ** dim
        start = self.start
        spacing = 0.5 * (self.end - start)
        sub_area_array = []
        for i in range(num_sub_areas):
            start_sub_area = np.zeros(dim)
            end_sub_area = np.zeros(dim)
            rest = i
            for d in reversed(list(range(dim))):
                start_sub_area[d] = start[d] + int(rest / 2 ** d) * spacing[d]
                end_sub_area[d] = start[d] + (int(rest / 2 ** d) + 1) * spacing[d]
                rest = rest % 2 ** d
            new_refinement_object = RefinementObjectExtendSplit(start_sub_area, end_sub_area, 1)
            sub_area_array.append(new_refinement_object)
        return sub_area_array


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
        newWidth = (self.end - self.start) / 2.0
        newObjects = []
        newObjects.append(RefinementObjectSingleDimension(self.start, self.start + newWidth, self.dim, coarsening_value))
        newObjects.append(RefinementObjectSingleDimension(self.start + newWidth, self.end, self.dim, coarsening_value))
        # self.finestWidth = min(newWidth,self.finestWidth)
        return newObjects, lmax_increase, update

    # in case lmax was changed the coarsening value of other RefinementObjects need to be increased
    def update(self, update_info):
        self.coarsening_level += update_info
