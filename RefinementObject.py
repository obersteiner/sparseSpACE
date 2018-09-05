
import math
import numpy as np
import abc,logging

# This class defines the general template of a Refinement Object that can be stored in the refinement container
class RefinementObject(object):
    def __init__(self, errorEstimator):
        self.errorEstimator = errorEstimator

    # set the local integral for area associated with RefinementObject
    def setIntegral(self, integral):
        self.integral = integral

    # refine this object and return newly created objects
    @abc.abstractmethod
    def refine(self):
        pass

    # set the local error associated with RefinementObject
    def setError(self, error):
        self.error = error


# This is the special class for the RefinementObject defined in the split extend scheme
class RefinementObjectExtendSplit(RefinementObject):
    def __init__(self, start, end, numberOfRefinementsBeforeExtend, coarseningValue=0, needExtendScheme=0):
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
        self.numberOfRefinementsBeforeExtend = numberOfRefinementsBeforeExtend
        self.evaluations = 0

    # this routine decides if we split or extend the RefinementObject
    def refine(self):
        coarseningLevel = self.coarseningValue
        if (
                self.needExtendScheme >= self.numberOfRefinementsBeforeExtend):  # add new component grids to scheme and refine only target area

            if self.coarseningValue == 0:
                self.coarseningValue = 0
            else:
                self.coarseningValue = coarseningLevel - 1
            # in case we have refined complete scheme (i.e. coarensingLevel was 0) we have to increase level everywhere else
            if (coarseningLevel == 0):
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
        elif (self.needExtendScheme >= 0):  # split the array
            # add to integralArray
            self.needExtendScheme += 1
            newRefinementObjects = self.splitAreaArbitraryDim()
            return newRefinementObjects, None, None
        else:
            print("Error!!!! Invalid value")
            assert (False)

    # in case lmax was changed the coarsening value of other RefinementObjects need to be increased
    def update(self, updateInfo):
        self.coarseningValue += updateInfo

    # splits the current area into 2**dim smaller ones and returns them
    def splitAreaArbitraryDim(self):
        dim = self.dim
        numSubAreas = 2 ** dim
        start = self.start
        spacing = 0.5 * (self.end - start)
        subAreaArray = []
        for i in range(numSubAreas):
            startSubArea = np.zeros(dim)
            endSubArea = np.zeros(dim)
            rest = i
            for d in reversed(list(range(dim))):
                startSubArea[d] = start[d] + int(rest / 2 ** d) * spacing[d]
                endSubArea[d] = start[d] + (int(rest / 2 ** d) + 1) * spacing[d]
                rest = rest % 2 ** d
            newRefinementObject = RefinementObjectExtendSplit(startSubArea, endSubArea,
                                                              self.numberOfRefinementsBeforeExtend,
                                                              self.coarseningValue, self.needExtendScheme)
            subAreaArray.append(newRefinementObject)
        return subAreaArray


# This is the special class for the RefinementObject defined in the split extend scheme
class RefinementObjectSingleDimension(RefinementObject):
    def __init__(self, start, end, dim, coarseningLevel=0):
        # start of subarea
        self.start = start
        # end of subarea
        self.end = end
        self.dim = dim
        # indicates how often area needs to be coarsened according to extend scheme
        self.coarseningLevel = coarseningLevel
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
        lmaxIncrease = None
        update = None
        coarseningValue = 0
        # if we are already at maximum refinement coarseningValue stays at 0
        if self.coarseningLevel == 0:
            coarseningValue = 0
        else:  # otherwise decrease coarsening level
            coarseningValue = self.coarseningLevel - 1
        # in case we have refined complete scheme (i.e. coarensingLevel was 0) we have to increase level everywhere else
        if (self.coarseningLevel == 0):  # extend scheme if we are at maximum refinement
            # increase coarsening level in all other dimensions
            for d in range(self.dim):
                for i in range(len(self.refinement[d])):
                    icoarsen = self.refinement[d][i][2] + 1
                    self.refinement[d][i] = (self.refinement[d][i][0], self.refinement[d][i][1], icoarsen)
            # increase lmax by 1
            lmaxIncrease = [1 for d in range(self.dim)]
            update = 1
            # print("New scheme")
            # self.scheme = getCombiScheme(self.lmin[0],self.lmax[0],self.dim)
            # self.newScheme = True
        # add new refined interval to refinement array (it has half of the width)
        newWidth = (self.end - self.start) / 2.0
        newObjects = []
        newObjects.append(RefinementObjectSingleDimension(self.start, self.start + newWidth, self.dim, coarseningValue))
        newObjects.append(RefinementObjectSingleDimension(self.start + newWidth, self.end, self.dim, coarseningValue))
        # self.finestWidth = min(newWidth,self.finestWidth)
        return newObjects, lmaxIncrease, update

    # in case lmax was changed the coarsening value of other RefinementObjects need to be increased
    def update(self, updateInfo):
        self.coarseningValue += updateInfo