import scipy.integrate
import numpy as np
import abc
import logging
from numpy import linalg as LA
from math import copysign

from sparseSpACE.Utils import LogUtility, print_levels, log_levels

# This class is the general interface of an error estimator currently used by the algorithm
class ErrorCalculator(object):
    # initialization
    def __init__(self, log_level: int = log_levels.WARNING, print_level: int = print_levels.NONE):
        self.log = logging.getLogger(__name__)
        self.is_global = False
        self.log_util = LogUtility(log_level=log_level, print_level=print_level)
        self.log_util.set_print_prefix('ErrorCalculator')
        self.log_util.set_log_prefix('ErrorCalculator')

    # calculates error for the function f and the integral information that was computed by the algorithm
    # this information contains the area specification and the approximated integral
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
    def calc_error(self, refine_object, norm, volume_weights=None):
        lower_bounds = np.array(refine_object.start)
        upper_bounds = np.array(refine_object.end)
        error = LA.norm(self.calc_area_error(refine_object.sub_integrals), norm)
        return max(error * np.prod(upper_bounds - lower_bounds))


class ErrorCalculatorExtendSplit(ErrorCalculator):
    def calc_error(self, refine_object, norm, volume_weights=None):
        if refine_object.switch_to_parent_estimation:
            return LA.norm(abs(refine_object.sum_siblings - refine_object.parent_info.previous_value), norm)
        else:
            return LA.norm(abs(refine_object.value - refine_object.parent_info.previous_value), norm)


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

class ErrorCalculatorSingleDimMisclassification(ErrorCalculator):
    def calc_error(self, refine_object, norm, volume_weights=None):
        volumes = refine_object.volume
        if volume_weights is None:
            #return LA.norm(abs(volumes), norm)
            return abs(volumes)
        # Normalized volumes
        #return LA.norm(abs(volumes * volume_weights), norm)
        return abs(volumes * volume_weights)

class ErrorCalculatorSingleDimMisclassificationGlobal(ErrorCalculator):
    def __init__(self):
        super().__init__()
        self.is_global = True

    def calc_error(self, refine_object, norm, volume_weights=None):
        volumes = refine_object.volume
        if volume_weights is None:
            #return LA.norm(abs(volumes), norm)
            return abs(volumes)
        # Normalized volumes
        #return LA.norm(abs(volumes * volume_weights), norm)
        return abs(volumes * volume_weights)

    def calc_global_error(self, data, grid_scheme):
        samples = data
        f = lambda x: grid_scheme(x)
        values = f(samples)
        for d in range(0, grid_scheme.dim):
            refinement_dim = grid_scheme.refinement.get_refinement_container_for_dim(d)
            for refinement_obj in refinement_dim.refinementObjects:
                # get the misclassification rate between start and end of refinement_obj
                hits = sum((1 for i in range(0, len(values))
                            if refinement_obj.start <= samples[i][d] <= refinement_obj.end
                            and copysign(1.0, values[i][0] == copysign(1.0, grid_scheme.operation.validation_classes[i]))))

                misses = sum((1 for i in range(0, len(values))
                              if refinement_obj.start <= samples[i][d] <= refinement_obj.end
                              and copysign(1.0, values[i][0]) != copysign(1.0, grid_scheme.operation.validation_classes[i])))

                if hits + misses > 0:
                    refinement_obj.add_volume(
                        np.array(misses * (refinement_obj.end - refinement_obj.start)))
                else:
                    # no data points were in this area
                    refinement_obj.add_volume(np.array(0.0))