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
    def get_total_num_points_arbitrary_dim(self, doNaive,
                                           distinctFunctionEvals=False):  # we assume here that all lmax entries are equal
        numpoints = 0
        for ss in self.scheme:
            num_sub_diagonal = (self.lmax[0] + self.dim - 1) - np.sum(ss[0])
            pointsgrid = self.get_num_points_arbitrary_dim(ss[0], doNaive, num_sub_diagonal)
            if distinctFunctionEvals:
                numpoints += pointsgrid * int(ss[1])
            else:
                numpoints += pointsgrid
        # print(numpoints)
        return numpoints

    # returns the number of points in a single component grid with refinement
    def get_num_points_arbitrary_dim(self, levelvec, do_naive, num_sub_diagonal):
        array2 = self.get_points_arbitrary_dim(levelvec, num_sub_diagonal)
        if do_naive:
            array2new = array2
        else:  # remove points that appear in the list multiple times
            array2new = list(set(array2))
        # print(len(array2new))
        return len(array2new)

    # prints every single component grid of the combination and orders them according to levels
    def print_resulting_combi_scheme(self, filename=None):
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
            num_sub_diagonal = (self.lmax[0] + dim - 1) - np.sum(ss[0])
            points = self.get_points_arbitrary_dim(ss[0], num_sub_diagonal)
            x_array = [p[0] for p in points]
            y_array = [p[1] for p in points]
            ax[lmax[1] - lmin[1] - (ss[0][1] - lmin[1]), (ss[0][0] - lmin[0])].plot(x_array, y_array, 'o', markersize=6,
                                                                                    color="black")
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
        plt.show()
        return fig

    # prints the sparse grid which results from the combination
    def print_resulting_sparsegrid(self, filename=None):
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
            points = self.get_points_arbitrary_dim(ss[0], numSubDiagonal)
            xArray = [p[0] for p in points]
            yArray = [p[1] for p in points]
            plt.plot(xArray, yArray, 'o', markersize=10, color="black")
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
        plt.show()
        return fig

    # check if combischeme is right
    def check_combi_scheme(self):
        if not self.grid.isNested():
            return
        dim = self.dim
        dictionary = {}
        for ss in self.scheme:
            num_sub_diagonal = (self.lmax[0] + dim - 1) - np.sum(ss[0])
            # print num_sub_diagonal , ii ,ss
            points = set(self.get_points_arbitrary_dim(ss[0], num_sub_diagonal))
            for p in points:
                if p in dictionary:
                    dictionary[p] += ss[1]
                else:
                    dictionary[p] = ss[1]
        # print(dictionary.items())
        for key, value in dictionary.items():
            # print(key, value)
            if value != 1:
                print("Failed for:", key, " with value: ", value)
                '''
                for area in self.refinement.getObjects():
                    print("new area:",area)
                    for ss in self.scheme:
                        num_sub_diagonal = (self.lmax[0] + dim - 1) - np.sum(ss[0])
                        self.coarsenGrid(ss[0],area, num_sub_diagonal,key)
                #print(self.refinement)
                #print(dictionary.items())
                '''
            assert (value == 1)

    def evaluate_final_combi(self, f):
        combiintegral = 0
        dim = self.dim
        # print "Dim:",dim
        num_evaluations = 0
        for ss in self.scheme:
            integral = 0
            for area in self.get_areas():
                area_integral, partial_integrals, evaluations = self.evaluate_area(f, area, ss[0])
                if area_integral != -2 ** 30:
                    num_evaluations += evaluations
                    integral += area_integral
            integral *= ss[1]
            combiintegral += integral
        return combiintegral, num_evaluations

    def init_adaptive_combi(self, f, minv, maxv, refinement_container, tol):
        self.tolerance = tol
        self.realIntegral = f.getAnalyticSolutionIntegral(self.a, self.b)
        if (refinement_container == []):  # initialize refinement
            self.lmin = [minv for i in range(self.dim)]
            self.lmax = [maxv for i in range(self.dim)]
            self.initialize_refinement()
        else:  # use the given refinement; in this case reuse old lmin and lmax and finestWidth; works only if there was no other run in between on same object
            self.refinement = refinement_container
            self.refinement.reinit_new_objects()
        # calculate the combination scheme
        self.scheme = getCombiScheme(self.lmin[0], self.lmax[0], self.dim)
        # initialize values
        self.refinements = 0
        # self.combiintegral = 0
        # self.subAreaIntegrals = []
        self.counter = 1
        # self.evaluationsTotal = 0 #number of evaluations in current grid
        # self.evaluationPerArea = [] #number of evaluations per area

    def evaluate_integral(self, f):
        # initialize values
        integralarrayComplete = []
        number_of_evaluations = 0
        # get tuples of all the combinations of refinement to access each subarea (this is the same for each component grid)
        areas = self.get_new_areas()
        # calculate integrals
        i = self.refinement.size() - self.refinement.new_objects_size()
        for area in areas:
            integralArrayIndividual = []
            evaluationsArea = 0
            for ss in self.scheme:  # iterate over component grids
                # initialize component grid specific variables

                numSubDiagonal = (self.lmax[0] + self.dim - 1) - np.sum(ss[0])
                integral = 0
                # iterate over all areas and calculate the integral

                area_integral, partial_integrals, evaluations = self.evaluate_area(f, area, ss[0])
                if area_integral != -2 ** 30:
                    number_of_evaluations += evaluations
                    if partial_integrals is not None:
                        integralArrayIndividual.extend(partial_integrals)
                    else:
                        integralArrayIndividual.append(ss[1] * area_integral)
                    # self.combiintegral += area_integral * ss[1]
                    evaluationsArea += evaluations
            self.refinement.set_integral(i, sum(integralArrayIndividual))
            self.refinement.set_evaluations(i, evaluationsArea / len(self.scheme))
            self.calc_error(i, f)
            i += 1
            # getArea with maximal error
        self.errorMax = self.refinement.get_max_error()
        print("max error:", self.errorMax)
        return abs(self.refinement.integral - self.realIntegral)

    def refine(self):
        # split all cells that have an error close to the max error
        areas = self.get_areas()
        self.prepare_refinement()
        self.refinement.clear_new_objects()
        margin = 0.9
        quit_refinement = False
        while True:  # refine all areas for which area is within margin
            # get next area that should be refined
            found_object, position, refine_object = self.refinement.get_next_object_for_refinement(
                tolerance=self.errorMax * margin)
            if found_object and not quit_refinement:  # new area found for refinement
                self.refinements += 1
                print("Refining position", position)
                quit_refinement = self.do_refinement(refine_object, position)

            else:  # all refinements done for this iteration -> reevaluate integral and check if further refinements necessary
                print("Finished refinement")
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

    # optimized adaptive refinement refine multiple cells in close range around max variance (here set to 10%)
    def performSpatiallyAdaptiv(self, minv, maxv, f, errorOperator, tol, refinement_container=[], do_plot=False):
        self.errorEstimator = errorOperator
        self.init_adaptive_combi(f, minv, maxv, refinement_container, tol)
        while (True):
            error = self.evaluate_integral(f)
            print("combiintegral:", self.refinement.integral)
            print("Current error:", error)
            # check if tolerance is already fullfilled with current refinement
            if error > tol:
                # refine further
                self.refine()
                if do_plot:
                    self.print_resulting_combi_scheme()
                    self.print_resulting_sparsegrid()
            else:  # refinement finished
                break
        # finished adaptive algorithm
        print("Number of refinements", self.refinements)
        self.check_combi_scheme()
        # evaluate final integral
        combiintegral, number_of_evaluations = self.evaluate_final_combi(f)
        return self.refinement, self.scheme, self.lmax, self.refinement.integral, number_of_evaluations

    @abc.abstractmethod
    def initialize_refinement(self):
        pass

    @abc.abstractmethod
    def get_points_arbitrary_dim(self, levelvec, numSubDiagonal):
        return

    @abc.abstractmethod
    def evaluate_area(self, f, area, levelvec):
        pass

    @abc.abstractmethod
    def do_refinement(self, area, position):
        pass

    # this is a default implementation that should be overritten if necessary
    def prepare_refinement(self):
        pass

    # this is a default implementation that should be overritten if necessary
    def refinement_postprocessing(self):
        self.refinement.apply_remove()

    # this is a default implementation that should be overritten if necessary
    def calc_error(self, objectID, f):
        self.refinement.calc_error(objectID, f)

    # this is a default implementation that should be overritten if necessary
    def get_new_areas(self):
        return self.refinement.get_new_objects()

    # this is a default implementation that should be overritten if necessary
    def get_areas(self):
        return self.refinement.get_objects()