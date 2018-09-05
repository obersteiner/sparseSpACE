import scipy.integrate
import numpy as np
import abc
import logging


# This class is the general interface of an error estimator currently used by the algorithm
class ErrorCalculator(object):
    # initialization
    def __init__(self):
        self.log = logging.getLogger(__name__)

    # calculates error for the function f and the integral information that was computed by the algorithm
    # this information contians the area specification and the approximated integral
    # current form is (approxIntegral,start,end)
    @abc.abstractmethod
    def calc_error(self, f, refine_object):
        return


# This error estimator doea a comparison to analytic solution. It outputs the absolut error.
class ErrorCalculatorAnalytic(ErrorCalculator):
    def calc_error(self, f, refine_object):
        lower_bounds = refine_object.start
        upper_bounds = refine_object.end
        real_integral_value = f.getAnalyticSolutionIntegral(lower_bounds, upper_bounds)
        return abs(refine_object.integral - real_integral_value)


# This error estimator doea a comparison to analytic solution. It outputs the relative error.
class ErrorCalculatorAnalyticRelative(ErrorCalculator):
    def calc_error(self, f, refine_object):
        lower_bounds = refine_object.start
        upper_bounds = refine_object.end
        real_integral_value = f.getAnalyticSolutionIntegral(lower_bounds, upper_bounds)
        real_integral_complete = f.getAnalyticSolutionIntegral(a, b)
        return abs((refine_object.integral - real_integral_value) / real_integral_complete)
