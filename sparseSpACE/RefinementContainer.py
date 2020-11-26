
import math
import numpy as np
from sparseSpACE.ErrorCalculator import ErrorCalculator
import abc,logging
from operator import itemgetter, attrgetter, methodcaller
from typing import Callable, Tuple, Sequence, List
from sparseSpACE.RefinementObject import RefinementObject
from sparseSpACE.Function import Function

# This class implements a general container that can be filled with refinementObjects (typically specified by the refinement strategy)
# In addition it stores accumulated values over all refinementObjects (like integral, numberOfEvaluations)
class RefinementContainer(object):
    def __init__(self, initial_objects: List[RefinementObject], dim: int, error_estimator: ErrorCalculator):
        self.refinementObjects = initial_objects
        self.dim = dim
        self.evaluationstotal = 0
        self.value = 0
        self.popArray = []
        self.startNewObjects = 0
        self.errorEstimator = error_estimator
        self.searchPosition = 0

    # returns the error that is associated with the specified refinementObject
    def get_error(self, object_id: int):
        return self.refinementObjects[object_id].error

    # refines the specified refinementObject
    def refine(self, object_id: int):
        # at first refinement in current refinement round we have
        # to save where the new RefinementObjects start
        if self.startNewObjects == 0:
            self.startNewObjects = len(self.refinementObjects)
        # refine RefinementObject;
        # returns the new RefinementObjects a possible update to lmax and update information for other RefinementObjects
        new_objects, lmax_update, update_information = self.refinementObjects[object_id].refine()
        # update other RefinementObjects if necessary
        if update_information is not None:
            self.update_values(update_information)
        # remove refined (and now outdated) RefinementObject
        self.prepare_remove(object_id)
        # add new RefinementObjects
        self.add(new_objects)
        return lmax_update, new_objects

    def update_values(self, update_info) -> None:
        for r in self.refinementObjects:
            r.update(update_info)

    def refinement_postprocessing(self) -> None:
        self.searchPosition = 0

    # if strategy decides from outside to update elements this function can be used
    def update_objects(self, update_info) -> None:
        for r in self.refinementObjects:
            r.update(update_info)

    # reset everything so that all RefinementObjects will be iterated
    def reinit_new_objects(self) -> None:
        self.startNewObjects = 0
        self.value = 0
        self.evaluationstotal = 0
        for obj in self.refinementObjects:
            obj.reinit()


    # return the maximal error among all RefinementObjects
    def get_max_error(self) -> float:
        max_error = 0
        for i in self.refinementObjects:
            if i.error > max_error:
                max_error = i.error
        return max_error

    # return the maximal benefit among all RefinementObjects
    def get_max_benefit(self) -> float:
        max_benefit = 0
        for i in self.refinementObjects:
            if i.benefit > max_benefit:
                max_benefit = i.benefit
        return max_benefit

    # return the maximal error among all RefinementObjects
    def get_total_error(self) -> float:
        total_error = 0
        for i in self.refinementObjects:
            total_error += i.error
        return total_error

    # indicate that all objects have been processed and new RefinementObjects will be added at the end
    def clear_new_objects(self) -> None:
        self.startNewObjects = len(self.refinementObjects)

        # returns only newly added RefinementObjects

    def get_new_objects(self) -> Sequence[RefinementObject]:
        return self.refinementObjects[self.startNewObjects:]

    # returns amount of newly added RefinementObjects
    def new_objects_size(self) -> int:
        return len(self.refinementObjects) - self.startNewObjects

    # prepares removing a RefinementObject (will be removed after refinement round)
    def prepare_remove(self, objectID) -> None:
        self.popArray.append(objectID)

    # remove all RefinementObjects that are outdated from container
    def apply_remove(self, sort=False) -> List[RefinementObject]:
        removed_objects = []
        for position in reversed(sorted(self.popArray)):
            self.value -= self.refinementObjects[position].value
            self.evaluationstotal -= self.refinementObjects[position].evaluations
            removed_object = self.refinementObjects.pop(position)
            removed_objects.append(removed_object)
            if self.startNewObjects != 0:
                self.startNewObjects -= 1
        self.popArray = []
        if sort:
            # sorted after every remove
            self.refinementObjects = sorted(self.refinementObjects, key=attrgetter('start'))
        return removed_objects

    # add new RefinementObjects to the container
    def add(self, new_refinement_objects) -> None:
        self.refinementObjects.extend(new_refinement_objects)

    # calculate the error according to the error estimator for specified RefinementObjects
    def calc_error(self, object_id, norm, volume_weights=None) -> None:
        refine_object = self.refinementObjects[object_id]
        refine_object.set_error(self.errorEstimator.calc_error(refine_object, norm, volume_weights=volume_weights))

    # returns all RefinementObjects in the container
    def get_objects(self) -> List[RefinementObject]:
        return self.refinementObjects

    def get_next_object_for_refinement(self, tolerance: float) -> Tuple[bool, int, RefinementObject]:
        if self.startNewObjects == 0:
            end = self.size()
        else:
            end = self.startNewObjects
        for i in range(self.searchPosition, end):
            if self.refinementObjects[i].benefit >= tolerance:
                self.searchPosition = i + 1
                return True, i, self.refinementObjects[i]
        return False, None, None

    # returns the specified RefinementObject from container
    def get_object(self, object_id: int) -> RefinementObject:
        return self.refinementObjects[object_id]

    # returns amount of RefinementObjects in container
    def size(self) -> int:
        return len(self.refinementObjects)

    # sets the number of evaluations associated with specified RefinementObject
    def set_evaluations(self, object_id: int, evaluations: int) -> None:
        # add evaluations also to total number of evaluations
        self.evaluationstotal += evaluations
        self.refinementObjects[object_id].set_evaluations(evaluations)

    def add_volume_of_children(self, object_id: int, volume: float, f) -> None:

        self.get_object(object_id).add_volume(volume, f)

        # self.calc_error(object_id, f)

        # print("setVolumeOfChildren not implemented yet", parameter)

    def printContainer(self) -> None:

        print("refineCont with this_dim:", self.dim)

        for obj in self.refinementObjects:
            obj.print()

    # sets the integral for area associated with specified RefinementObject
    def set_value(self, object_id: int, value: Sequence[float]) -> None:
        # also add integral to global integral value
        self.value += value
        self.refinementObjects[object_id].set_value(value)

    def set_benefit(self, object_id: int) -> None:
        refine_object = self.refinementObjects[object_id]
        if refine_object.evaluations != 0:
            refine_object.benefit = refine_object.error / refine_object.evaluations
        else:
            refine_object.benefit = refine_object.error

    def get_max_coarsening(self) -> int:
        max_coarsening_level = 0
        for refine_obj in self.refinementObjects:
            max_coarsening_level = max(max_coarsening_level, refine_obj.coarsening_level)
        return max_coarsening_level

# this class defines a container of refinement containers for each dimension in the single dimension test case
# it delegates methods to subcontainers and coordinates everything
class MetaRefinementContainer(object):
    def __init__(self, refinement_containers: Sequence[RefinementContainer], calculate_volume_weights=False):
        self.refinementContainers = refinement_containers
        self.evaluationstotal = 0
        self.value = [0]
        # for get next obj for refinement:
        self.curContainer = 0
        self.calculate_volume_weights = calculate_volume_weights

    # return the maximal error among all RefinementContainers
    def get_max_benefit(self) -> float:
        max_benefit = 0.0
        for c in self.refinementContainers:
            benefit = c.get_max_benefit()
            # print("getMaxERror", error)
            if max_benefit < benefit:
                max_benefit = benefit
        return max_benefit

    def get_max_coarsening(self, d: int) -> int:
        return self.refinementContainers[d].get_max_coarsening()

    def get_object(self, object_position: Tuple[int, int]) -> RefinementObject:
        return self.refinementContainers[object_position[0]].get_object(object_position[1])

    # return the maximal error among all RefinementContainers
    def get_total_error(self) -> float:
        total_error = 0.0
        for c in self.refinementContainers:
            total_error += c.get_total_error()
        return total_error

    def refinement_postprocessing(self) -> None:
        for c in self.refinementContainers:
            c.refinement_postprocessing()

    # sets the value for area associated with whole meta container
    def set_value(self, objectID: int, value: Sequence[float]) -> None:
        assert(False)
        self.value = value

    # sets the number of evaluations associated with whole meta container
    def set_evaluations(self, objectID: int, evaluations: int) -> None:
        self.evaluationstotal = evaluations

    # delegate to containers
    def reinit_new_objects(self) -> None:
        self.curContainer = 0
        for c in self.refinementContainers:
            c.reinit_new_objects()

    def size(self) -> int:
        return 1

    def new_objects_size(self) -> int:
        return 1

    def clear_new_objects(self) -> None:
        for container in self.refinementContainers:
            container.clear_new_objects()

    def get_next_object_for_refinement(self, tolerance: float) -> Tuple[bool, Tuple[int, int], RefinementObject]:
        foundObj = False
        if self.curContainer == len(self.refinementContainers):
            return False, None, None
        while foundObj == False:
            result, index, obj = self.refinementContainers[self.curContainer].get_next_object_for_refinement(tolerance)
            if (result == False):
                self.curContainer += 1
                if (self.curContainer == len(self.refinementContainers)):
                    return False, None, None
                # check current obj
            else:
                foundObj = True
                return True, (self.curContainer, index), obj
        print("get_next_object_for_refinement: should not end here!")
        return False, None, None

    # delegate to containers
    def apply_remove(self, sort: bool=False) -> List[RefinementObject]:
        for c in self.refinementContainers:
            c.apply_remove(sort)
        return []

    def get_refinement_container_for_dim(self, d: int) -> RefinementContainer:
        """This method returns the RefinementContainer of dimension d.

        :param d: Dimension of requested RefinementContainer.
        :return: RefinementContainer
        """
        return self.refinementContainers[d]

    # apply refinement
    def refine(self, position: Tuple[int, int]) -> Sequence[int]:
        # position[0]: which container(=dimension), position[1]: which object(=area)
        # self.updateGridSpecificRefinementInfo(position)
        new_lmax_change, new_objects = self.refinementContainers[position[0]].refine(position[1])
        #if new_lmax_change is not None:
        #    for d, c in enumerate(self.refinementContainers):
        #        if d != position[0]:
        #            c.update_objects(new_lmax_change[d])
        # self.print()
        return new_lmax_change, new_objects

    # calculate the error according to the error estimator for specified RefinementObjects
    def calc_error(self, object_id: int, norm: int) -> None:
        if self.calculate_volume_weights:
            volume_weights = np.array([1.0 / v if abs(v) > 10 ** -10 else 1.0 for v in self.value])
        else:
            volume_weights = None
        for cont in self.refinementContainers:
            for obj in range(0, cont.size()):
                cont.calc_error(obj, norm, volume_weights=volume_weights)

    # calculate the error according to the error estimator for specified RefinementObjects
    def set_benefit(self, object_id: int) -> None:
        for cont in self.refinementContainers:
            for obj in range(0, cont.size()):
                cont.set_benefit(obj)

    def set_dictChildren(self, dictChildren):
        self.dictChildren = dictChildren

    def set_dictCoordinate(self, dictCoordinate):
        self.dictCoordinate = dictCoordinate

    def set_depthLevelVector(self, depthLevelVector):
        self.depthLevelVector = depthLevelVector

    # for testing purposes:
    def print(self) -> None:
        print("--------------------")
        print("printMetaRefinement")
        print("--------------------")
        # print("dictCoord:", self.dictCoordinate)
        # print("dictChildren:", self.dictChildren)
        # print("depthLevelVector:", self.depthLevelVector)
        self.print_containers_only()

    def print_containers_only(self) -> None:
        for cont in self.refinementContainers:
            print("-")
            cont.printContainer()

    def print_size(self) -> None:
        print("----------------\nRefinement Information sizes")
        for d in range(0, len(self.refinementContainers)):
            print("Dim: ", d, "\t", self.get_refinement_container_for_dim(d).size())

'''
class RefinementContainerCell(RefinementContainer):
    def __init__(self, initial_objects, dim, error_estimator, a, b):
        super().__init__(self, initial_objects, dim, error_estimator)
        self.grid_dictionary = {}
        index_set = CombiScheme.get_index_set()
        for object in initial_objects:
            if tuple(object.levelvec) in self.grid_dictionary:
                self.grid_dictionary[tuple(object.levelvec)].append((object.start,object.end))
            else:
                self.grid_dictionary[tuple(object.levelvec)] = [(object.start,object.end)]

    def refinement_postprocessing(self, new_objects, object_id):
        for object in new_objects:
            previous_levelvec = [max(CombiScheme.lmin,object.levelvec - 1) for d in range(self.dim)]

            self.grid_dictionary[tuple(object.levelvec)] = self.grid_dictionary[tuple(previous_levelvec)]

            area_info_old = (self.grid_dictionary[tuple(self.getObject(object_id).levelvec)].start, self.grid_dictionary[tuple(self.getObject(object_id).levelvec)].end)
            self.grid_dictionary[tuple(object.levelvec)].remove(area_info_old)

        for object in new_objects:
            self.grid_dictionary[tuple(object.levelvec)].append((object.start, object.end))

    def set_integral(self, object_id, integral):
        self.integral += integral

    def set_evaluations(self, object_id, evaluations):
        self.evaluationstotal += evaluations

    def calc_error(self, object_id, f):
        pass
'''


