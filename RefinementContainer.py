
import math
import numpy as np
import ErrorCalculator
import abc,logging


# This class implements a general container that can be filled with refinementObjects (typically specified by the refinement strategy)
# In addition it stores accumulated values over all refinementObjects (like integral, numberOfEvaluations)
class RefinementContainer(object):
    def __init__(self, initialObjects, dim, errorEstimator):
        self.refinementObjects = initialObjects
        self.dim = dim
        self.evaluationstotal = 0
        self.integral = 0
        self.popArray = []
        self.startNewObjects = 0
        self.errorEstimator = errorEstimator
        self.searchPosition = 0

    # returns the error that is associated with the specified refinementObject
    def getError(self, objectID):
        return self.refinementObjects[objectID].error

    # refines the specified refinementObject
    def refine(self, objectID):
        # at first refinement in current refinement round we have
        # to save where the new RefinementObjects start
        if (self.startNewObjects == 0):
            self.startNewObjects = len(self.refinementObjects)
        # refine RefinementObject;
        # returns the new RefinementObjects a possible update to lmax and update information for other RefinementObjects
        newObjects, lmaxUpdate, updateInformation = self.refinementObjects[objectID].refine()
        # update other RefinementObjects if necessary
        if updateInformation != None:
            for r in self.refinementObjects:
                r.update(updateInformation)
        # remove refined (and now outdated) RefinementObject
        self.prepareRemove(objectID)
        # add new RefinementObjects
        self.add(newObjects)
        return lmaxUpdate

    # if strategy decides from outside to update elements this function can be used
    def updateObjects(self, updateInfo):
        for r in self.refinementObjects:
            r.update(updateInfo)

    # reset everything so that all RefinementObjects will be iterated
    def reinitNewObjects(self):
        self.startNewObjects = 0
        self.integral = 0
        self.evaluationstotal = 0

    # return the maximal error among all RefinementObjects
    def getMaxError(self):
        maxError = 0
        for i in self.refinementObjects:
            if (i.error > maxError):
                maxError = i.error
        return maxError

    # indicate that all objects have been processed and new RefinementObjects will be added at the end
    def clearNewObjects(self):
        self.startNewObjects = len(self.refinementObjects)

        # returns only newly added RefinementObjects

    def getNewObjects(self):
        return self.refinementObjects[self.startNewObjects:]

    # returns amount of newly added RefinementObjects
    def newObjectsSize(self):
        return len(self.refinementObjects) - self.startNewObjects

    # prepares removing a RefinementObject (will be removed after refinement round)
    def prepareRemove(self, objectID):
        self.popArray.append(objectID)

    # remove all RefinementObjects that are outdated from container
    def applyRemove(self):
        for position in reversed(sorted(self.popArray)):
            self.integral -= self.refinementObjects[position].integral
            self.evaluationstotal -= self.refinementObjects[position].evaluations
            self.refinementObjects.pop(position)
            if (self.startNewObjects != 0):
                self.startNewObjects -= 1
        self.popArray = []
        self.searchPosition = 0

    # add new RefinementObjects to the container
    def add(self, newRefinementObjects):
        self.refinementObjects.extend(newRefinementObjects)

    # calculate the error according to the error estimator for specified RefinementObjects
    def calcError(self, objectID, f):
        refineObject = self.refinementObjects[objectID]
        refineObject.setError(self.errorEstimator.calcError(f, refineObject))

    # returns all RefinementObjects in the container
    def getObjects(self):
        return self.refinementObjects

    def getNextObjectForRefinement(self, tolerance):
        if (self.startNewObjects == 0):
            end = self.size()
        else:
            end = self.startNewObjects
        for i in range(self.searchPosition, end):
            if (self.refinementObjects[i].error >= tolerance):
                self.searchPosition = i + 1
                return True, i, self.refinementObjects[i]
        return False, None, None

    # returns the specified RefinementObject from container
    def getObject(self, objectID):
        return self.refinementObjects[objectID]

    # returns amount of RefinementObjects in container
    def size(self):
        return len(self.refinementObjects)

    # sets the number of evaluations associated with specified RefinementObject
    def setEvaluations(self, objectID, evaluations):
        # add evaluations also to total number of evaluations
        self.evaluationstotal += evaluations
        self.refinementObjects[objectID].evaluations = evaluations

    # sets the integral for area associated with specified RefinementObject
    def setIntegral(self, objectID, integral):
        # also add integral to global integral value
        self.integral += integral
        self.refinementObjects[objectID].setIntegral(integral)


# this class defines a container of refinement containers for each dimension in the single dimension test case
# it delegates methods to subcontainers and coordinates everything
class MetaRefinementContainer(object):
    def __init__(self, refinementContainers):
        self.refinementContainers = refinementContainers

    # return the maximal error among all RefinementContainers
    def getMaxError(self):
        maxError = 0
        for c in self.refinementContainers:
            error = c.getMaxError()
            if (maxError < error):
                maxError = error
        return maxError

    # sets the integral for area associated with whole meta container
    def setIntegral(self, objectID, integral):
        self.integral = integral

    # sets the number of evaluations associated with whole meta container
    def setEvaluations(self, objectID, evaluations):
        self.evaluations = evaluations

    # delegate to containers
    def reinitNewObjects(self):
        for c in self.refinementContainers:
            c.reinitNewObjects()

    def size(self):
        return 1

    def newObjectsSize(self):
        return 1

    def clearNewObjects(self):
        pass

    def getNextObjectForRefinement(self, tolerance):
        pass
        # toDo

    # delegate to containers
    def applyRemove(self):
        for c in self.refinementContainers:
            c.applyRemove()

    def getRefinementContainerForDim(self, d):
        return self.refinementContainers[d]

    # apply refinement
    def refine(self, position):
        lmaxChange = self.refinementContainers[position[0]].refine(position[1])
        if (lmaxChange != None):
            for d, c in enumerate(self.refinementContainers):
                if (d != position[0]):
                    c.update(1)
        return lmaxChange
