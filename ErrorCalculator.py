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

from RefinementObject import RefinementObjectCell

# This error estimator doea a comparison to analytic solution. It outputs the absolut error.
class ErrorCalculatorAnalyticCell(ErrorCalculator):
    def calc_error(self, f, refine_object):
        lower_bounds = refine_object.start
        upper_bounds = refine_object.end
        real_integral_value = f.getAnalyticSolutionIntegral(lower_bounds, upper_bounds)
        #integral = refine_object.integral
        error = self.calc_area_error(refine_object.sub_integrals, real_integral_value)
        '''
        for d in range(refine_object.dim):
            levelvec_copy = list(refine_object.levelvec)
            levelvec_copy[d] += 1
            for child in RefinementObjectCell.children_cell_arbitrary_dim(d, refine_object.start, refine_object.end, refine_object.dim):
                for parent in RefinementObjectCell.get_parents(levelvec_copy, child[0], child[1], refine_object.a, refine_object.b, refine_object.dim):
                    if parent in RefinementObjectCell.cell_dict:
                        new_error = self.calc_area_error(RefinementObjectCell.cell_dict[parent].sub_integrals, real_integral_value)
                        if new_error > error:
                            error = new_error
        #print(refine_object.get_key(), integral, real_integral_value, abs(integral - real_integral_value))
        '''
        return error

    def calc_area_error(self, sub_integrals, real_integral_value):
        error = 0
        for sub_integral in sub_integrals:
            error += sub_integral[0] * sub_integral[1]
        return abs(error)