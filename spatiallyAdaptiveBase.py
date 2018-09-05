import abc,logging
# Python modules
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import operator
import numpy as np
import scipy as sp
import scipy.integrate
from scipy.interpolate import interpn
import scipy.special
import math
import time
from RefinementContainer import *
from RefinementObject import *
from combiScheme import *
from Grid import *
from ErrorCalculator import *

# This class defines the general interface and functionalties of all spatially adaptive refinement strategies
class SpatiallyAdaptivBase(object):
    def __init__(self, a, b, grid=TrapezoidalGrid()):
        self.log = logging.getLogger(__name__)
        self.dim = len(a)
        self.a = a
        self.b = b
        self.grid = grid
        assert (len(a) == len(b))

    # calculate the total number of points used in the complete combination scheme
    def getTotalNumPointsArbitraryDim(self, doNaive,
                                      distinctFunctionEvals=False):  # we assume here that all lmax entries are equal
        numpoints = 0
        for ss in self.scheme:
            numSubDiagonal = (self.lmax[0] + self.dim - 1) - np.sum(ss[0])
            pointsgrid = self.getNumPointsArbitraryDim(ss[0], doNaive, numSubDiagonal)
            if distinctFunctionEvals:
                numpoints += pointsgrid * int(ss[1])
            else:
                numpoints += pointsgrid
        # print(numpoints)
        return numpoints

    # returns the number of points in a single component grid with refinement
    def getNumPointsArbitraryDim(self, levelvec, doNaive, numSubDiagonal):
        array2 = self.getPointsArbitraryDim(levelvec, numSubDiagonal)
        if (doNaive):
            array2new = array2
        else:  # remove points that appear in the list multiple times
            array2new = list(set(array2))
        # print(len(array2new))
        return len(array2new)

    # prints every single component grid of the combination and orders them according to levels
    def printResultingCombiScheme(self, filename=None):
        plt.rcParams.update({'font.size': 22})
        scheme = self.scheme
        lmin = self.lmin
        lmax = self.lmax
        dim = self.dim
        if dim != 2:
            print("Cannot print combischeme of dimension > 2")
            return None
        fig, ax = plt.subplots(ncols=lmax[0] - lmin[0] + 1, nrows=lmax[1] - lmin[1] + 1, figsize=(20, 20))
        # get points of each component grid and plot them individually
        for i in range(lmax[0] - lmin[0] + 1):
            for j in range(lmax[1] - lmin[1] + 1):
                ax[i, j].xaxis.set_ticks_position('none')
                ax[i, j].yaxis.set_ticks_position('none')
                ax[i, j].set_xlim([self.a[0] - 0.005, self.b[0] + 0.005])
                ax[i, j].set_ylim([self.a[1] - 0.005, self.b[1] + 0.005])
        for ss in scheme:
            numSubDiagonal = (self.lmax[0] + dim - 1) - np.sum(ss[0])
            points = self.getPointsArbitraryDim(ss[0], numSubDiagonal)
            xArray = [p[0] for p in points]
            yArray = [p[1] for p in points]
            ax[lmax[1] - lmin[1] - (ss[0][1] - lmin[1]), (ss[0][0] - lmin[0])].plot(xArray, yArray, 'o', markersize=6,
                                                                                    color="black")
        if (filename != None):
            plt.savefig(filename, bbox_inches='tight')
        plt.show()
        return fig

    # prints the sparse grid which results from the combination
    def printResultingSparsegrid(self, filename=None):
        plt.rcParams.update({'font.size': 32})
        scheme = self.scheme
        dim = self.dim
        if dim != 2:
            print("Cannot print sparse grid of dimension > 2")
            return None
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.set_xlim([self.a[0] - 0.01, self.b[0] + 0.01])
        ax.set_ylim([self.a[1] - 0.01, self.b[1] + 0.01])
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        # ax.axis('off')
        # get points of each component grid and plot them in one plot
        for ss in scheme:
            numSubDiagonal = (self.lmax[0] + dim - 1) - np.sum(ss[0])
            points = self.getPointsArbitraryDim(ss[0], numSubDiagonal)
            xArray = [p[0] for p in points]
            yArray = [p[1] for p in points]
            plt.plot(xArray, yArray, 'o', markersize=10, color="black")
        if (filename != None):
            plt.savefig(filename, bbox_inches='tight')
        plt.show()
        return fig

    # check if combischeme is right
    def checkCombiScheme(self):
        if self.grid.isNested() == False:
            return
        dim = self.dim
        dictionary = {}
        for ss in self.scheme:
            numSubDiagonal = (self.lmax[0] + dim - 1) - np.sum(ss[0])
            # print numSubDiagonal , ii ,ss
            points = set(self.getPointsArbitraryDim(ss[0], numSubDiagonal))
            for p in points:
                if (p in dictionary):
                    dictionary[p] += ss[1]
                else:
                    dictionary[p] = ss[1]
        # print(dictionary.items())
        for key, value in dictionary.items():
            # print(key, value)
            if (value != 1):
                print("Failed for:", key, " with value: ", value)
                '''
                for area in self.refinement.getObjects():
                    print("new area:",area)
                    for ss in self.scheme:
                        numSubDiagonal = (self.lmax[0] + dim - 1) - np.sum(ss[0])
                        self.coarsenGrid(ss[0],area, numSubDiagonal,key)
                #print(self.refinement)
                #print(dictionary.items())
                '''
            assert (value == 1)

    def evaluateFinalCombi(self, f):
        combiintegral = 0
        dim = self.dim
        # print "Dim:",dim
        numEvaluations = 0
        for ss in self.scheme:
            integral = 0
            for area in self.getAreas():
                areaIntegral, partialIntegrals, evaluations = self.evaluateArea(f, area, ss[0])
                if (areaIntegral != -2 ** 30):
                    numEvaluations += evaluations
                    integral += areaIntegral
            integral *= ss[1]
            combiintegral += integral
        return combiintegral, numEvaluations

    def initAdaptiveCombi(self, f, minv, maxv, refinementContainer, tol):
        self.tolerance = tol
        self.realIntegral = f.getAnalyticSolutionIntegral(self.a, self.b)
        if (refinementContainer == []):  # initialize refinement
            self.lmin = [minv for i in range(self.dim)]
            self.lmax = [maxv for i in range(self.dim)]
            self.initializeRefinement()
        else:  # use the given refinement; in this case reuse old lmin and lmax and finestWidth; works only if there was no other run in between on same object
            self.refinement = refinementContainer
            self.refinement.reinitNewObjects()
        # calculate the combination scheme
        self.scheme = getCombiScheme(self.lmin[0], self.lmax[0], self.dim)
        # initialize values
        self.refinements = 0
        # self.combiintegral = 0
        # self.subAreaIntegrals = []
        self.counter = 1
        # self.evaluationsTotal = 0 #number of evaluations in current grid
        # self.evaluationPerArea = [] #number of evaluations per area

    def evaluateIntegral(self, f):
        # initialize values
        integralarrayComplete = []
        numberOfEvaluations = 0
        # get tuples of all the combinations of refinement to access each subarea (this is the same for each component grid)
        areas = self.getNewAreas()
        # calculate integrals
        i = self.refinement.size() - self.refinement.newObjectsSize()
        for area in areas:
            integralArrayIndividual = []
            evaluationsArea = 0
            for ss in self.scheme:  # iterate over component grids
                # initialize component grid specific variables

                numSubDiagonal = (self.lmax[0] + self.dim - 1) - np.sum(ss[0])
                integral = 0
                # iterate over all areas and calculate the integral

                areaIntegral, partialIntegrals, evaluations = self.evaluateArea(f, area, ss[0])
                if (areaIntegral != -2 ** 30):
                    numberOfEvaluations += evaluations
                    if (partialIntegrals != None):
                        integralArrayIndividual.extend(partialIntegrals)
                    else:
                        integralArrayIndividual.append(ss[1] * areaIntegral)
                    # self.combiintegral += areaIntegral * ss[1]
                    evaluationsArea += evaluations
            self.refinement.setIntegral(i, sum(integralArrayIndividual))
            self.refinement.setEvaluations(i, evaluationsArea / len(self.scheme))
            self.calcError(i, f)
            i += 1
            # getArea with maximal error
        self.errorMax = self.refinement.getMaxError()
        print("max error:", self.errorMax)
        return abs(self.refinement.integral - self.realIntegral)

    def refine(self):
        # split all cells that have an error close to the max error
        areas = self.getAreas()
        self.prepareRefinement()
        self.refinement.clearNewObjects()
        margin = 0.9
        quitRefinement = False
        while (True):  # refine all areas for which area is within margin
            # get next area that should be refined
            foundObject, position, refineObject = self.refinement.getNextObjectForRefinement(
                tolerance=self.errorMax * margin)
            if (foundObject and not (quitRefinement)):  # new area found for refinement
                self.refinements += 1
                print("Refining position", position)
                quitRefinement = self.doRefinement(refineObject, position)

            else:  # all refinements done for this iteration -> reevaluate integral and check if further refinements necessary
                print("Finished refinement")
                self.refinementPostprocessing()
                break
        if (self.refinements / 100 > self.counter):
            self.refinement.reinitNewObjects()
            self.combiintegral = 0
            self.subAreaIntegrals = []
            self.evaluationPerArea = []
            self.evaluationsTotal = 0
            self.counter += 1
            print("recalculating errors")

    # optimized adaptive refinement refine multiple cells in close range around max variance (here set to 10%)
    def performSpatiallyAdaptiv(self, minv, maxv, f, errorOperator, tol, refinementContainer=[], plot=False):
        self.errorEstimator = errorOperator
        self.initAdaptiveCombi(f, minv, maxv, refinementContainer, tol)
        while (True):
            error = self.evaluateIntegral(f)
            print("combiintegral:", self.refinement.integral)
            print("Current error:", error)
            # check if tolerance is already fullfilled with current refinement
            if (error > tol):
                # refine further
                self.refine()
                if (plot == True):
                    self.printResultingCombiScheme()
                    self.printResultingSparsegrid()
            else:  # refinement finished
                break
        # finished adaptive algorithm
        print("Number of refinements", self.refinements)
        self.checkCombiScheme()
        # evaluate final integral
        combiintegral, numberOfEvaluations = self.evaluateFinalCombi(f)
        return self.refinement, self.scheme, self.lmax, self.refinement.integral, numberOfEvaluations

    @abc.abstractmethod
    def initializeRefinement(self):
        pass

    @abc.abstractmethod
    def getPointsArbitraryDim(self, levelvec, numSubDiagonal):
        return

    @abc.abstractmethod
    def evaluateArea(self, f, area, levelvec):
        pass

    @abc.abstractmethod
    def doRefinement(self, area, position):
        pass

    # this is a default implementation that should be overritten if necessary
    def prepareRefinement(self):
        pass

    # this is a default implementation that should be overritten if necessary
    def refinementPostprocessing(self):
        self.refinement.applyRemove()

    # this is a default implementation that should be overritten if necessary
    def calcError(self, objectID, f):
        self.refinement.calcError(objectID, f)

    # this is a default implementation that should be overritten if necessary
    def getNewAreas(self):
        return self.refinement.getNewObjects()

    # this is a default implementation that should be overritten if necessary
    def getAreas(self):
        return self.refinement.getObjects()