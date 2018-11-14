import numpy as np
import combiScheme
import Grid
import math
import logging
from combiScheme import *
from Grid import *


# T his class implements the standard combination technique
class DimAdaptiveCombi(object):
    # initialization
    # a = lower bound of integral; b = upper bound of integral
    # grid = specified grid (e.g. Trapezoidal);
    def __init__(self, a, b, grid=None):
        self.log = logging.getLogger(__name__)
        self.dim = len(a)
        self.a = a
        self.b = b
        self.grid = grid
        self.combischeme = CombiScheme(self.dim)
        assert (len(a) == len(b))

    # standard combination scheme for quadrature
    # lmin = minimum level; lmax = target level
    # f = function to integrate; dim=dimension of problem
    def perform_combi(self, minv, maxv, f, dim, tolerance):
        start = self.a
        end = self.b
        # compute minimum and target level vector
        lmin = [minv for i in range(dim)]
        lmax = [maxv for i in range(dim)]
        real_integral = f.getAnalyticSolutionIntegral(self.a, self.b)

        self.combischeme.init_adaptive_combi_scheme(maxv, minv)
        combiintegral = 0
        while (abs(combiintegral - real_integral)/abs(real_integral)) > tolerance:
            combiintegral = 0
            self.scheme = self.combischeme.getCombiScheme(lmin[0], lmax[0], dim)
            error_array = np.zeros(len(self.scheme))
            for i, ss in enumerate(self.scheme):
                integral = self.grid.integrate(f, ss[0], start, end) * ss[1]
                # as error estimator we compare to the analytic solution and divide by the cost=number of points in grid
                error_array[i] = abs(integral - real_integral)/abs(real_integral) / np.prod(self.grid.levelToNumPoints(ss[0])) if self.combischeme.is_refinable(ss[0]) else 0
                combiintegral += integral
            grid_id = np.argmax(error_array)
            print(error_array)
            print("Current error:", abs(combiintegral - real_integral)/abs(real_integral))
            print("Refining", self.scheme[grid_id], self.combischeme.is_refinable(self.scheme[grid_id][0]))
            self.combischeme.update_adaptive_combi(self.scheme[grid_id][0])
        print("CombiSolution", combiintegral)
        print("Analytic Solution", real_integral)
        print("Difference", abs(combiintegral - real_integral))
        return self.scheme, abs(combiintegral - real_integral), combiintegral

    # calculate the number of points for a standard combination scheme
    def get_total_num_points(self, distinct_function_evals=False):
        num_points = 0
        for ss in self.scheme:
            if distinct_function_evals and self.grid.isNested():
                factor = int(ss[1])
            else:
                factor = 1
            self.grid.setCurrentArea(self.a, self.b, ss[0])
            num_points_array = np.array(self.grid.levelToNumPoints(ss[0]))
            num_points += np.prod(num_points_array) * factor
        return num_points
