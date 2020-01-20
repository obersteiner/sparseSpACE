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
    def calc_error(self, refine_object, norm):
        return

# This error estimator does a surplus estimation. It outputs the absolute error.
class ErrorCalculatorSurplusCell(ErrorCalculator):
    def calc_error(self, refine_object, norm, volume_weights=None):
        error = LA.norm(self.calc_area_error(refine_object.sub_integrals), norm)
        return error

    def calc_area_error(self, sub_integrals):
        error = 0.0
        for sub_integral in sub_integrals:
            error += sub_integral[0] * sub_integral[1]
        return abs(error)


class ErrorCalculatorSurplusCellPunishDepth(ErrorCalculatorSurplusCell):
    def calc_error(self, refine_object, volume_weights=None):
        lower_bounds = np.array(refine_object.start)
        upper_bounds = np.array(refine_object.end)
        error = self.calc_area_error(refine_object.sub_integrals)
        return max(error * np.prod(upper_bounds - lower_bounds))


class ErrorCalculatorExtendSplit(ErrorCalculator):
    def calc_error(self, refine_object, norm, volume_weights=None):
        if refine_object.switch_to_parent_estimation:
            return LA.norm(abs(refine_object.sum_siblings - refine_object.parent_info.previous_value), norm)
        else:
            return LA.norm(abs(refine_object.integral - refine_object.parent_info.previous_value), norm)


class ErrorCalculatorSingleDimVolumeGuided(ErrorCalculator):
    def calc_error(self, refine_object, norm, volume_weights=None):
        # pagoda-volume
        volumes = refine_object.volume
        if volume_weights is None:
            return LA.norm(abs(volumes), norm)
        # Normalized volumes
        return LA.norm(abs(volumes * volume_weights), norm)


class ErrorCalculatorSingleDimVolumeGuidedPunishedDepth(ErrorCalculator):
    def calc_error(self, refineObj, norm):
        #width of refineObj:
        width = refineObj.end - refineObj.start
        # pagoda-volume
        volume = LA.norm(refineObj.volume * (width), norm)
        return abs(volume)
