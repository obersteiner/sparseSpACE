import scipy.integrate
import numpy as np
import abc
import logging
from numpy import linalg as LA

# This class is the general interface of an error estimator currently used by the algorithm
class ErrorCalculator(object):
    # initialization
    def __init__(self):
        self.log = logging.getLogger(__name__)

    # calculates error for the function f and the integral information that was computed by the algorithm
    # this information contians the area specification and the approximated integral
    # current form is (approxIntegral,start,end)
    @abc.abstractmethod
    def calc_error(self, f, refine_object, norm):
        return


# This error estimator doea a comparison to analytic solution. It outputs the absolute error.
class ErrorCalculatorAnalytic(ErrorCalculator):
    def calc_error(self, f, refine_object, norm):
        lower_bounds = refine_object.start
        upper_bounds = refine_object.end
        real_integral_value = f.getAnalyticSolutionIntegral(lower_bounds, upper_bounds)
        return LA.norm(abs(refine_object.integral - real_integral_value), norm)


# This error estimator doea a comparison to analytic solution. It outputs the relative error.
class ErrorCalculatorAnalyticRelative(ErrorCalculator):
    def calc_error(self, f, refine_object, norm):
        lower_bounds = refine_object.start
        upper_bounds = refine_object.end
        real_integral_value = f.getAnalyticSolutionIntegral(lower_bounds, upper_bounds)
        real_integral_complete = f.getAnalyticSolutionIntegral(a, b)
        return LA.norm(abs((refine_object.integral - real_integral_value) / real_integral_complete), norm)


# This error estimator does a surplus estimation. It outputs the absolute error.
class ErrorCalculatorSurplusCell(ErrorCalculator):
    def calc_error(self, f, refine_object, norm):
        error = LA.norm(self.calc_area_error(refine_object.sub_integrals), norm)
        return error

    def calc_area_error(self, sub_integrals):
        error = 0.0
        for sub_integral in sub_integrals:
            error += sub_integral[0] * sub_integral[1]
        return abs(error)


class ErrorCalculatorSurplusCellPunishDepth(ErrorCalculatorSurplusCell):
    def calc_error(self, f, refine_object):
        lower_bounds = np.array(refine_object.start)
        upper_bounds = np.array(refine_object.end)
        error = self.calc_area_error(refine_object.sub_integrals)
        return max(error * np.prod(upper_bounds - lower_bounds))


class ErrorCalculatorExtendSplit(ErrorCalculator):
    def calc_error(self, f, refine_object, norm):
        if refine_object.switch_to_parent_estimation:
            return LA.norm(abs(refine_object.sum_siblings - refine_object.parent_info.previous_value), norm)
        else:
            return LA.norm(abs(refine_object.integral - refine_object.parent_info.previous_value), norm)


class ErrorCalculatorSingleDimVolumeGuided(ErrorCalculator):
    def calc_error(self, f, refineObj, norm):
        # pagoda-volume
        volume = refineObj.volume
        return LA.norm(abs(volume), norm)


class ErrorCalculatorSingleDimVolumeGuidedPunishedDepth(ErrorCalculator):
    def calc_error(self, f, refineObj, norm):
        #width of refineObj:
        width = refineObj.end - refineObj.start
        # pagoda-volume
        volume = LA.norm(refineObj.volume * (width), norm)
        return abs(volume)
