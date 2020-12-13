import numpy as np
import matplotlib.pyplot as plt
import warnings
import time
import random as rnd
from sparseSpACE.StandardCombi import StandardCombi
from sparseSpACE.GridOperation import DensityEstimation
from sklearn import datasets, preprocessing, neighbors
from sklearn.utils import shuffle
from typing import List, Tuple, Union, Iterable

from sparseSpACE.ErrorCalculator import ErrorCalculator, ErrorCalculatorSingleDimVolumeGuided
from sparseSpACE.Grid import GlobalTrapezoidalGrid
from sparseSpACE.spatiallyAdaptiveSingleDimension2 import SpatiallyAdaptiveSingleDimensions2
from sparseSpACE.Utils import log_levels, print_levels, logUtil, LogUtility

import cProfile
import pstats
import os


class DataSet:
    """Type of data sets on which to perform DensityEstimation, Classification and Clustering.

    All DataSets contain data in the form of a tuple of length 2 with one ndarray each:
    The samples in arbitrary dimension and the corresponding labels in dimension 1.
    Unknown labels are labeled with -1.
    """

    def __init__(self, raw_data: Union[Tuple[np.ndarray, ...], np.ndarray, str], name: str = 'unknown', label: str = 'class',
                 log_level: int = log_levels.WARNING, print_level: int = print_levels.NONE):
        """Constructor of the DataSet class.

        Takes raw data and optionally a name or label-description as parameter and initializes the original data in the form of a tuple of legnth 2.
        Scaling attributes are unassigned until scaling occurs.

        :param raw_data: Samples (and corresponding labels) of this DataSet. Can be a tuple of samples and labels, only labelless samples, CSV file.
        :param name: Optional. Name of this DataSet.
        :param label: Optional. Type of labels for this DataSet.
        :param log_level: Optional. Set the log level for this instance. Only statements of the given level or higher will be written to the log file
        :param print_level: Optional.  Set the level for print statements. Only statements of the given level or higher will be written to the console
        """
        self._name = name
        self._label = label
        self._data = None
        self._dim = None
        self._shape = None
        self._shuffled = False
        self._scaled = False
        self._scaling_range = None
        self._scaling_factor = None
        self._original_min = None
        self._original_max = None
        self.log_util = LogUtility(log_level=log_level, print_level=print_level)
        self.log_util.set_print_prefix('DataSet')
        self.log_util.set_log_prefix('DataSet')
        self._initialize(raw_data)
        assert ((self._data is not None) and (self._dim is not None) and (self._shape is not None))
        assert (isinstance(self._data, tuple) and len(self._data) == 2 and
                isinstance(self._data[0], np.ndarray) and isinstance(self._data[1], np.ndarray))

    def __getitem__(self, item: int) -> np.ndarray:
        return self._data[item]

    def __str__(self) -> str:
        return str(self._data)

    def copy(self) -> 'DataSet':
        copied = DataSet(self._data)
        copied.__dict__.update(self.__dict__)
        copied.set_name("%s_copy" % self.get_name())
        return copied

    def set_name(self, name: str) -> None:
        self._name = name

    def set_label(self, label: str) -> None:
        self._label = label

    def get_name(self) -> str:
        return self._name

    def get_label(self) -> str:
        return self._label

    def get_data(self) -> Tuple[np.ndarray, ...]:
        return self._data

    def get_min_data(self) -> Union[np.ndarray, None]:
        if not self.is_empty():
            return np.amin(self._data[0], axis=0)
        else:
            return None

    def get_max_data(self) -> Union[np.ndarray, None]:
        if not self.is_empty():
            return np.amax(self._data[0], axis=0)
        else:
            return None

    def get_original_min(self) -> Union[np.ndarray, None]:
        return self._original_min

    def get_original_max(self) -> Union[np.ndarray, None]:
        return self._original_max

    def get_length(self) -> int:
        length = round(self._data[0].size / self._dim) if (self._dim != 0) else 0
        assert ((length * self._dim) == self._data[0].size)
        return length

    def get_dim(self) -> int:
        return self._dim

    def get_number_labels(self) -> int:
        return len([x for x in set(self._data[1]) if x >= 0])

    def get_labels(self) -> List[int]:
        return list(set(self._data[1]))

    def has_labelless_samples(self) -> bool:
        return -1 in self._data[1]

    def is_empty(self) -> bool:
        return self._data[0].size == 0

    def is_shuffled(self) -> bool:
        return self._shuffled

    def is_scaled(self) -> bool:
        return self._scaled

    def get_scaling_range(self) -> Tuple[float, float]:
        return self._scaling_range

    def get_scaling_factor(self) -> float:
        return self._scaling_factor

    def _initialize(self, raw_data: Union[Tuple[np.ndarray, np.ndarray], np.ndarray]) -> None:
        """Initialization method for the DataSet class.

        Provides several checks of the input parameter raw_data passed by the constructor and raises an error if raw_data can't be converted to an
        appropriate form.

        :param raw_data: Samples (and corresponding labels) of this DataSet. Can be a tuple of samples and labels or only labelless samples.
        :return: None
        """
        if isinstance(raw_data, np.ndarray):
            if raw_data.size == 0:
                self._dim = 0
                self._shape = 0
                raw_data = np.reshape(raw_data, 0)
            else:
                self._dim = round(raw_data.size / len(raw_data))
                self._shape = (len(raw_data), self._dim)
                assert ((len(raw_data) * self._dim) == raw_data.size)
                raw_data = np.reshape(raw_data, self._shape)
            self._data = raw_data, np.array(([-1] * len(raw_data)), dtype=np.int64)
        elif isinstance(raw_data, tuple) and (len(raw_data) == 2):
            if isinstance(raw_data[0], list):
                raise ValueError("Invalid raw_data parameter in DataSet Constructor.")
            if raw_data[0].size == 0:
                self._dim = 0
                self._shape = 0
                self._data = tuple([np.reshape(raw_data[0], 0), raw_data[1]])
            elif raw_data[1].ndim == 1 and not any([x < -1 for x in list(set(raw_data[1]))]) and (len(raw_data[0]) == len(raw_data[1])):
                self._dim = round(raw_data[0].size / len(raw_data[0]))
                self._shape = (len(raw_data[0]), self._dim)
                assert ((len(raw_data[0]) * self._dim) == raw_data[0].size)
                self._data = tuple([np.reshape(raw_data[0], self._shape), raw_data[1]])
            else:
                raise ValueError("Invalid raw_data parameter in DataSet Constructor.")
        else:
            raise ValueError("Invalid raw_data parameter in DataSet Constructor.")

    def _update_internal(self, to_update: 'DataSet') -> 'DataSet':
        """Update all internal attributes, which can normally only be changed through DataSet methods or should be changed automatically.

        Mainly used to keep scaling of DataSets after methods that change the internal data.

        :param to_update: DataSet, whose internal attributes need to updated.
        :return: Input DataSet with updated internal attributes.
        """
        to_update._label = self._label
        to_update._shuffled = self._shuffled
        to_update._scaled = self._scaled
        to_update._scaling_range = self._scaling_range
        to_update._scaling_factor = self._scaling_factor
        if self._scaled:
            to_update._original_min = self._original_min.copy()
            to_update._original_max = self._original_max.copy()
        else:
            to_update._original_min = self._original_min
            to_update._original_max = self._original_max
        return to_update

    def same_scaling(self, to_check: 'DataSet') -> bool:
        """Check, whether self and to_check have the same scaling.

        Compares the scaling range and factor of self and to_check and returns False if anything doesn't match.

        :param to_check: DataSet, whose internal scaling should be compared to self's internal scaling.
        :return: Boolean value, which indicates whether the internal scaling of input DataSet and self are completely equal.
        """
        if not self._scaled == to_check._scaled:
            return False
        if not self._scaled and not to_check._scaled:
            return True
        assert (self._scaled == to_check._scaled)
        if not (isinstance(self._scaling_range[0], Iterable) != isinstance(to_check._scaling_range[0], Iterable)):
            if isinstance(self._scaling_range[0], Iterable):
                scaling_range = all([(x[0] == y[0]) and (x[1] == y[1]) for x, y in zip(self._scaling_range, to_check._scaling_range)])
            else:
                scaling_range = all([x == y for x, y in zip(self._scaling_range, to_check._scaling_range)])
        else:
            return False
        if not (isinstance(self._scaling_factor, Iterable) != isinstance(to_check._scaling_factor, Iterable)):
            if isinstance(self._scaling_factor, Iterable):
                scaling_factor = all([x == y for x, y in zip(self._scaling_factor, to_check._scaling_factor)])
            else:
                scaling_factor = self._scaling_factor == to_check._scaling_factor
        else:
            return False
        return scaling_range and scaling_factor

    def remove_samples(self, indices: List[int]) -> 'DataSet':
        """Remove samples of a DataSet object at specified indices.

        If the list of indices is empty, no samples are removed.

        :param indices: List of indices at which to remove samples.
        :return: New DataSet in which samples are removed.
        """
        if any([(i < 0) or (i > self.get_length()) for i in indices]):
            raise ValueError("Can't remove samples out of bounds of DataSet.")
        removed_samples = [self._update_internal(DataSet((np.array([self._data[0][i]]), np.array([self._data[1][i]])))) for i in indices]
        self._data = tuple([np.delete(self._data[0], indices, axis=0), np.delete(self._data[1], indices, axis=0)])
        return DataSet.list_concatenate(removed_samples)

    def scale_range(self, scaling_range: Tuple[float, float], override_scaling: bool = False) -> None:
        """Scale DataSet data to a specified range.

        If override_scaling is set, current scaling (if available) is turned into the original scaling of this DataSet and the new scaling
        specified by the input parameter is applied.

        :param scaling_range: Range to which all samples should be scaled.
        :param override_scaling: Optional. Conditional parameter, which indicates whether old scaling (if available) should be overridden.
        :return: None
        """
        if not self._scaled or override_scaling:
            scaler = preprocessing.MinMaxScaler(feature_range=scaling_range)
            scaler.fit(self._data[0])
            self._data = tuple([scaler.transform(self._data[0]), np.array([c for c in self._data[1]])])
            self._scaled = True
            self._scaling_range = scaling_range
            self._scaling_factor = scaler.scale_
            self._original_min = scaler.data_min_
            self._original_max = scaler.data_max_
        else:
            scaler = preprocessing.MinMaxScaler(feature_range=scaling_range)
            scaler.fit(self._data[0])
            self._data = tuple([scaler.transform(self._data[0]), np.array([c for c in self._data[1]])])
            self._scaling_range = scaling_range
            self._scaling_factor *= scaler.scale_

    def scale_factor(self, scaling_factor: Union[float, np.ndarray], override_scaling: bool = False) -> None:
        """Scale DataSet by a specified factor.

        If override_scaling is set, current scaling (if available) is turned into the original scaling of this DataSet and the new scaling
        specified by the input parameter is applied.

        :param scaling_factor: Factor by which all samples should be scaled. Can either be a float value for general scaling or np.ndarray with
        dimension self._dim to scale each dimension individually.
        :param override_scaling: Optional. Conditional parameter, which indicates whether old scaling (if available) should be overridden.
        :return: None
        """
        if not self._scaled or override_scaling:
            self._original_min = self.get_min_data()
            self._original_max = self.get_max_data()
            self._data = tuple([np.array(list(map(lambda x: x * scaling_factor, self._data[0]))), self._data[1]])
            self._scaling_range = (np.amin(self._data[0], axis=0), np.amax(self._data[0], axis=0))
            self._scaling_factor = scaling_factor
            self._scaled = True
        else:
            if isinstance(scaling_factor, np.ndarray) and len(scaling_factor) != self._dim:
                raise ValueError("Multidimensional scaling factor needs to have the same dimension as DataSet it is applied to.")
            self._data = tuple([np.array(list(map(lambda x: x * scaling_factor, self._data[0]))), self._data[1]])
            self._scaling_range = (np.amin(self._data[0], axis=0), np.amax(self._data[0], axis=0))
            self._scaling_factor *= scaling_factor

    def shift_value(self, shift_val: Union[float, np.ndarray], override_scaling: bool = False) -> None:
        """Shift DataSet by a specified value.

        If override_scaling is set, current scaling (if available) is turned into the original scaling of this DataSet and the new scaling
        specified by the input parameter is applied.

        :param shift_val: Value by which all samples should be shifted. Can either be a float value for general shifting or np.ndarray with
        dimension self._dim to shift each dimension individually.
        :param override_scaling: Optional. Conditional parameter, which indicates whether old scaling (if available) should be overridden.
        :return: None
        """
        if not self._scaled or override_scaling:
            self._original_min = self.get_min_data()
            self._original_max = self.get_max_data()
            self._data = tuple([np.array(list(map(lambda x: (x + shift_val), self._data[0]))), self._data[1]])
            self._scaling_range = (np.amin(self._data[0], axis=0), np.amax(self._data[0], axis=0))
            self._scaling_factor = 1.0
            self._scaled = True
        else:
            if isinstance(shift_val, np.ndarray) and len(shift_val) != self._dim:
                raise ValueError("Multidimensional shifting value needs to have the same dimension as DataSet it is applied to.")
            self._data = tuple([np.array(list(map(lambda x: (x + shift_val), self._data[0]))), self._data[1]])
            self._scaling_range = (np.amin(self._data[0], axis=0), np.amax(self._data[0], axis=0))

    def shuffle(self) -> None:
        """Shuffle data samples of DataSet object randomly.

        :return: None
        """
        shuffled = shuffle(tuple(zip(self._data[0], self._data[1])))
        self._data = tuple([np.array([[v for v in x[0]] for x in shuffled]), np.array([y[1] for y in shuffled])])
        self._shuffled = True

    def concatenate(self, other_dataset: 'DataSet') -> 'DataSet':
        """Concatenate this DataSet's data with the data of a specified DataSet.

        If either this or the specified DataSet are empty, the other one is returned.
        Only data of DataSets with equal dimension can be concatenated.

        :param other_dataset: DataSet, whose data should be concatenated with data of this DataSet.
        :return: New DataSet with concatenated data.
        """
        if not (self._dim == other_dataset.get_dim()):
            if other_dataset.is_empty():
                return self
            elif self.is_empty():
                return other_dataset
            else:
                raise ValueError("DataSets must have the same dimensions for concatenation.")
        values = np.concatenate((self._data[0], other_dataset[0]), axis=0)
        labels = np.concatenate((self._data[1], other_dataset[1]))
        concatenated_set = DataSet((values, labels))
        self._update_internal(concatenated_set)
        equal_scaling = self.same_scaling(concatenated_set)
        if not equal_scaling:
            raise ValueError("Can't concatenate DataSets with different scaling")
        return concatenated_set

    @staticmethod
    def list_concatenate(list_datasets: List['DataSet']) -> 'DataSet':
        """Concatenate a list of DataSets to a single DataSet.

        If an empty list is received as a parameter, an empty DataSet is returned.

        :param list_datasets: List of DataSet's which to concatenate.
        :return: New DataSet, which contains the concatenated data of all DataSets within the list.
        """
        if len(list_datasets) == 0:
            return DataSet(tuple([np.array([]), np.array([])]))
        dataset = list_datasets[0]
        for i in range(1, len(list_datasets)):
            dataset = dataset.concatenate(list_datasets[i])
        return dataset

    def split_labels(self) -> List['DataSet']:
        """Split samples with the same labels into their own DataSets.

        Creates a DataSet for each label and puts all samples with corresponding label into this DataSet.
        Stores all of those single-label-DataSets into a list.

        :return: A List, which contains all single-label-DataSets.
        """
        set_labels = []
        for j in self.get_labels():
            current_values = np.array([x for i, x in enumerate(self._data[0]) if self._data[1][i] == j])
            current_label = np.array(([j] * len(current_values)), dtype=np.int64)
            current_set = DataSet(tuple([current_values, current_label]))
            self._update_internal(current_set)
            set_labels.append(current_set)
        return set_labels

    def split_one_vs_others(self) -> List['DataSet']:
        """Separate samples into two classes: the class to be estimated and all others combined into the 'other' set

        Collects the for each label that is not the one to be estimated into one set.
        The label for the 'other' set is a weighted negative number. The weight is the ratio of samples between
        the class to be estimated and all others.

        :return: A List, which contains the two DataSets.
        """
        set_classes = []
        class_numbers = [sum(self._data[1] == j) for j in self.get_labels()]
        for j in self.get_labels():
            values = np.array([x for i, x in enumerate(self._data[0])])
            others = (sum(class_numbers) - class_numbers[j])
            labels = np.array([1 if self._data[1][i] == j else max(-1, -1 * (class_numbers[j] / others)) for i, x in
                               enumerate(self._data[0])])
            current_set = DataSet(tuple([values, labels]))
            self._update_internal(current_set)
            set_classes.append(current_set)
        return set_classes

    def split_without_labels(self) -> Tuple['DataSet', 'DataSet']:
        """Separates samples without from samples with labels.

        Creates a DataSet for samples with and without labels each.
        Samples are stored in the respective DataSet.
        If there are no labelless samples and/ or samples with labels, the respective DataSet stays empty.

        :return: A Tuple of two new DataSets, which contain all labelless samples and all samples with labels.
        """
        labelless_values = np.array([x for i, x in enumerate(self._data[0]) if self._data[1][i] == -1])
        labelfull_values = np.array([x for i, x in enumerate(self._data[0]) if self._data[1][i] >= 0])
        set_labelless = DataSet(labelless_values)
        set_labelfull = DataSet(tuple([labelfull_values, np.array([c for c in self._data[1] if c >= 0], dtype=np.int64)]))
        self._update_internal(set_labelless)
        self._update_internal(set_labelfull)
        return set_labelless, set_labelfull

    def split_pieces(self, percentage: float) -> Tuple['DataSet', 'DataSet']:
        """Splits this DataSet's data into two pieces specified by the percentage parameter.

        The first split piece contains all samples until index (percentage * this data's length) rounded down to the next integer.
        The second split piece contains all other samples.
        Before the splitting is performed, percentage is checked if in range (0, 1) and if not is set to 1.

        :param percentage: Percentage of this DataSet's data at whose last index the split occurs.
        :return: A Tuple of two new DataSets, which contain all samples before and after the index at which the data was split.
        """
        percentage = percentage if 0 <= percentage < 1 else 1.0
        set0 = DataSet(tuple([np.array(self._data[0][:(round(self.get_length() * percentage))]),
                              self._data[1][:(round(self.get_length() * percentage))]]))
        set1 = DataSet(tuple([np.array(self._data[0][(round(self.get_length() * percentage)):]),
                              self._data[1][(round(self.get_length() * percentage)):]]))
        self._update_internal(set0)
        self._update_internal(set1)
        return set0, set1

    def remove_labels(self, percentage: float) -> None:
        """Removes the labels of percentage samples randomly.

        Before removal of labels, percentage is checked if in range (0, 1) and if not is set to 1.
        Mainly used for testing purposes.

        :param percentage: Percentage of random indices at which to remove labels.
        :return: None
        """
        labelless, labelfull = self.split_without_labels()
        indices = rnd.sample(range(0, labelfull.get_length()), round((percentage if (1 > percentage >= 0) else 1.0) * labelfull.get_length()))
        labels = labelfull._data[1]
        labels[indices] = -1
        if labelless.is_empty():
            self._data = tuple([labelfull._data[0], labels])
        elif labelfull.is_empty():
            self._data = tuple([labelless._data[0], labelless._data[1]])
        else:
            self._data = tuple([np.concatenate((labelfull._data[0], labelless._data[0])), np.concatenate((labels, labelless._data[1]))])

    def move_boundaries_to_front(self) -> None:
        """Move samples with lowest and highest value in each dimension to the front of this DataSet's data.

        Mainly used for initialization in Classification to guarantee the boundary samples being in the learning DataSet.

        :return: None
        """
        search_indices_min = np.where(self._data[0] == self.get_min_data())
        search_indices_max = np.where(self._data[0] == self.get_max_data())
        indices = list(set(search_indices_min[0]) | set(search_indices_max[0]))
        for i, x in enumerate(indices):
            self._data[0][[i, x]] = self._data[0][[x, i]]
            self._data[1][[i, x]] = self._data[1][[x, i]]

    def revert_scaling(self) -> None:
        """Revert the scaling of this DataSet's data to its original scaling.

        Scaling is applied in reverse to this DataSet's data and scaling attributes are returned to their initial setting.

        :return:
        """
        self.scale_factor(1.0 / self._scaling_factor, override_scaling=False)
        self.shift_value(-(self.get_min_data() - self._original_min), override_scaling=False)
        self._scaled = False
        self._scaling_range = None
        self._scaling_factor = None
        self._original_min = None
        self._original_max = None

    def density_estimation(self,
                           masslumping: bool = True,
                           lambd: float = 0.0,
                           minimum_level: int = 1,
                           maximum_level: int = 5,
                           one_vs_others: bool = False,
                           reuse_old_values: bool = False,
                           pre_scaled_data: bool = False,
                           plot_de_dataset: bool = True,
                           plot_density_estimation: bool = True,
                           plot_combi_scheme: bool = True,
                           plot_sparsegrid: bool = True) -> Tuple[StandardCombi, DensityEstimation]:
        """Perform the GridOperation DensityEstimation on this DataSet.

        This method can also plot the DensityEstimation results directly.
        For more information on DensityEstimation, please refer to the DensityEstimation class in the GridOperation module.

        :param masslumping: Optional. Conditional Parameter, which indicates whether masslumping should be enabled for DensityEstimation.
        :param lambd: Optional. Parameter, which adjusts the 'smoothness' of DensityEstimation results.
        :param minimum_level: Optional. Minimum Level of Sparse Grids on which to perform DensityEstimation.
        :param maximum_level: Optional. Maximum Level of Sparse Grids on which to perform DensityEstimation.
        :param maximum_level: Optional. Maximum Level of Sparse Grids on which to perform DensityEstimation
        :param one_vs_others: Optional. Data from other classes will be included with a weighted label < 0.
        :param reuse_old_values: Optional. R-values and b-values will be re-used across refinements and component grids.
        :param pre_scaled_data: Optional. Deactivates data scaling in the grid operation.
        :param plot_de_dataset: Optional. Conditional Parameter, which indicates whether this DataSet should be plotted for DensityEstimation.
        :param plot_density_estimation: Optional. Conditional Parameter, which indicates whether results of DensityEstimation should be plotted.
        :param plot_combi_scheme: Optional. Conditional Parameter, which indicates whether resulting combi scheme of DensityEstimation should be
        plotted.
        :param plot_sparsegrid: Optional. Conditional Parameter, which indicates whether resulting sparsegrid of DensityEstimation should be plotted.
        :return: Tuple of the resulting StandardCombi and DensityEstimation objects.
        """
        a = np.zeros(self._dim)
        b = np.ones(self._dim)

        if one_vs_others:
            de_object = DensityEstimation(self._data, self._dim, masslumping=masslumping, lambd=lambd,
                                          reuse_old_values=reuse_old_values, classes=self._data[1],
                                          pre_scaled_data=pre_scaled_data, print_output=False)
        else:
            de_object = DensityEstimation(self._data, self._dim, masslumping=masslumping, lambd=lambd,
                                          reuse_old_values=reuse_old_values, pre_scaled_data=pre_scaled_data,
                                          print_output=False)

        combi_object = StandardCombi(a, b, operation=de_object, print_output=False)

        # For profiling:
        # pStuff.increment_class_counter()
        # cls = '__class-' + str(pStuff.get_class_counter())
        # profileName = 'profiles/pStd_' + pStuff.get_data_set_used() + '_' + pStuff.get_file_name_extension()+cls
        # while os.path.isfile(profileName):
        #     profileName = profileName + '+'
        # cProfile.runctx('combi_object.perform_operation(minimum_level, maximum_level)', globals(), locals(), filename=profileName)
        # with open(profileName+'_TIME.txt', "w") as f:
        #     ps = pstats.Stats(profileName, stream=f)
        #     ps.sort_stats(pstats.SortKey.TIME)
        #     ps.print_stats()
        # with open(profileName+'_CUMU.txt', "w") as f:
        #     ps = pstats.Stats(profileName, stream=f)
        #     ps.sort_stats(pstats.SortKey.CUMULATIVE)
        #     ps.print_stats()
        # os.remove(profileName)

        combi_object.perform_operation(minimum_level, maximum_level)
        if plot_de_dataset:
            if de_object.scaled:
                self.scale_range((0, 1), override_scaling=True)
            self.plot(plot_labels=False)
        if plot_density_estimation:
            combi_object.plot(contour=True)
        if plot_combi_scheme:
            combi_object.print_resulting_combi_scheme(operation=de_object)
        if plot_sparsegrid:
            combi_object.print_resulting_sparsegrid(markersize=20)
        return combi_object, de_object

    def density_estimation_dimension_wise(self,
                                          masslumping: bool = True,
                                          lambd: float = 0.0,
                                          minimum_level: int = 1,
                                          maximum_level: int = 5,
                                          reuse_old_values: bool = False,
                                          numeric_calculation: bool = True,
                                          margin: float = 0.5,
                                          tolerance: float = 0.01,
                                          max_evaluations: int = 256,
                                          rebalancing: bool = False,
                                          modified_basis: bool = False,
                                          boundary: bool = False,
                                          one_vs_others: bool = True,
                                          error_calculator: ErrorCalculator = None,
                                          pre_scaled_data: bool = False,
                                          single_step_refinement: bool = False,
                                          plot_de_dataset: bool = True,
                                          plot_density_estimation: bool = True,
                                          plot_combi_scheme: bool = True,
                                          plot_sparsegrid: bool = True) -> Tuple[StandardCombi, DensityEstimation]:
        """Perform the GridOperation DensityEstimation dimension-wise on this DataSet.

        This method can also plot the DensityEstimation results directly.
        For more information on DensityEstimation, please refer to the DensityEstimation class in the GridOperation module.

        :param masslumping: Optional. Conditional Parameter, which indicates whether masslumping should be enabled for DensityEstimation.
        :param lambd: Optional. Parameter, which adjusts the 'smoothness' of DensityEstimation results.
        :param minimum_level: Optional. Minimum Level of Sparse Grids on which to perform DensityEstimation.
        :param maximum_level: Optional. Maximum Level of Sparse Grids on which to perform DensityEstimation.
        :param reuse_old_values: Optional. R-values and b-values will be re-used across refinements and component grids.
        :param numeric_calculation: Optional. Use numerical calculation for the integral.
        :param margin: Optional.
        :param tolerance: Optional. Error tolerance. Refinement stops if this threshold is reached
        :param max_evaluations: Optional. Maximum number of points. The refinement will stop when it exceeds this limit.
        :param rebalancing: Optional. Activate rebalancing of refinement trees.
        :param modified_basis: Optional. Use the modified basis function..
        :param boundary: Optional. Put points on the boundary.
        :param one_vs_others: Optional. Data from other classes will be included with a weighted label < 0.
        :param error_calculator: Optional. Explicitly pass the error calculator that you want to use.
        :param pre_scaled_data: Optional. Deactivates data scaling in the grid operation.
        :param single_step_refinement: Optional. Regardless of max_evaluations or tolerance, only a single refinement step will be executed
        :param plot_de_dataset: Optional. Conditional Parameter, which indicates whether this DataSet should be plotted for DensityEstimation.
        :param plot_density_estimation: Optional. Conditional Parameter, which indicates whether results of DensityEstimation should be plotted.
        :param plot_combi_scheme: Optional. Conditional Parameter, which indicates whether resulting combi scheme of DensityEstimation should be
        plotted.
        :param plot_sparsegrid: Optional. Conditional Parameter, which indicates whether resulting sparsegrid of DensityEstimation should be plotted.
        :return: Tuple of the resulting StandardCombi and DensityEstimation objects.
        """
        a = np.zeros(self._dim)
        b = np.ones(self._dim)
        grid = GlobalTrapezoidalGrid(a=a, b=b, modified_basis=modified_basis, boundary=boundary)

        if error_calculator is None:
            error_calculator = ErrorCalculatorSingleDimVolumeGuided()

        classes = None
        if one_vs_others:
            classes = self._data[1]

        de_object = DensityEstimation(self._data,
                                      self._dim,
                                      grid=grid,
                                      masslumping=masslumping,
                                      lambd=lambd,
                                      classes=classes,
                                      reuse_old_values=reuse_old_values,
                                      numeric_calculation=numeric_calculation,
                                      print_output=False,
                                      pre_scaled_data=pre_scaled_data)
        combi_object = SpatiallyAdaptiveSingleDimensions2(a, b, operation=de_object, margin=margin, rebalancing=rebalancing)

        # pStuff.increment_class_counter()
        # cls = '__class-' + str(pStuff.get_class_counter())
        # profileName = 'profiles/pStd_' + pStuff.get_data_set_used() + '_' + pStuff.get_file_name_extension()+cls
        # while os.path.isfile(profileName):
        #     profileName = profileName + '+'
        # cProfile.runctx('combi_object.performSpatiallyAdaptiv(minimum_level,
        #                                                       maximum_level,
        #                                                       error_calculator,
        #                                                       tolerance,
        #                                                       max_evaluations=max_evaluations,
        #                                                       do_plot=plot_combi_scheme,
        #                                                       print_output=False)',
        #                 globals(),
        #                 locals(),
        #                 filename=profileName)
        # with open(profileName+'_TIME.txt', "w") as f:
        #     ps = pstats.Stats(profileName, stream=f)
        #     ps.sort_stats(pstats.SortKey.TIME)
        #     ps.print_stats()
        # with open(profileName+'_CUMU.txt', "w") as f:
        #     ps = pstats.Stats(profileName, stream=f)
        #     ps.sort_stats(pstats.SortKey.CUMULATIVE)
        #     ps.print_stats()
        # os.remove(profileName)

        combi_object.performSpatiallyAdaptiv(minimum_level, maximum_level, error_calculator, tolerance,
                                             max_evaluations=max_evaluations, do_plot=plot_combi_scheme,
                                             print_output=False,
                                             single_step=single_step_refinement)
        if plot_de_dataset:
            if de_object.scaled:
                self.scale_range((0, 1), override_scaling=True)
            self.plot(plot_labels=False)
        if plot_density_estimation:
            combi_object.plot(contour=True)
        if plot_combi_scheme:
            combi_object.print_resulting_combi_scheme(operation=de_object)
        if plot_sparsegrid:
            combi_object.print_resulting_sparsegrid(markersize=20)
        return combi_object, de_object

    def plot(self, plot_labels: bool = True, filename: str = None) -> plt.Figure:
        """Plot DataSet.

        Plotting is only available for dimensions 2 and 3.

        :param plot_labels: Optional. Conditional parameter, which indicates whether labels should be coloured for plotting.
        :param filename: Optional. pass a filename (e.g. /path/to/folder/myPlot.png) to save the figure
        :return: Figure, which is plotted.
        """
        plt.rc('font', size=30)
        plt.rc('axes', titlesize=40)
        plt.rc('figure', figsize=(12.0, 12.0))
        fig = plt.figure()

        if self._dim == 2:
            ax = fig.add_subplot(111)
            if plot_labels:
                if self.has_labelless_samples():
                    data_labelless, data_labelfull = self.split_without_labels()
                    list_labels = data_labelfull.split_labels()
                    x, y = zip(*data_labelless[0])
                    ax.scatter(x, y, s=125, label='%s_?' % self._label, c='gray')
                else:
                    list_labels = self.split_labels()
                for i, v in enumerate(list_labels):
                    x, y = zip(*v[0])
                    ax.scatter(x, y, s=125, label='%s_%d' % (self._label, i))
                ax.legend(fontsize=22, loc='upper left', borderaxespad=0.0, bbox_to_anchor=(1.05, 1))
            else:
                x, y = zip(*self._data[0])
                ax.scatter(x, y, s=125)
            ax.set_title(self._name)
            ax.title.set_position([0.5, 1.025])
            ax.grid(True)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        elif self._dim == 3:
            ax = fig.add_subplot(111, projection='3d')
            if plot_labels:
                if self.has_labelless_samples():
                    data_labelless, data_labelfull = self.split_without_labels()
                    list_labels = data_labelfull.split_labels()
                    x, y, z = zip(*data_labelless[0])
                    ax.scatter(x, y, z, s=125, label='%s_?' % self._label, c='gray')
                else:
                    list_labels = self.split_labels()
                for i, v in enumerate(list_labels):
                    x, y, z = zip(*v[0])
                    ax.scatter(x, y, z, s=125, label='%s_%d' % (self._label, i))
                ax.legend(fontsize=22, loc='upper left', borderaxespad=0.0, bbox_to_anchor=(1.05, 1))
            else:
                fig.set_figwidth(10.0)
                x, y, z = zip(*self._data[0])
                ax.scatter(x, y, z, s=125)
            ax.set_title(self._name)
            ax.title.set_position([0.5, 1.025])
            ax.grid(True)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
        else:
            warnings.formatwarning = lambda msg, ctg, fname, lineno, file=None, line=None: "%s:%s: %s: %s\n" % (fname, lineno, ctg.__name__, msg)
            warnings.warn("Invalid dimension for plotting. Couldn't plot DataSet.", stacklevel=3)

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
        plt.show()
        return fig


class Classification:
    """Type of objects, that classify data based on some previously performed learning.
    """

    def __init__(self,
                 raw_data: 'DataSet',
                 data_range: Tuple[np.ndarray, np.ndarray] = None,
                 split_percentage: float = 1.0,
                 split_evenly: bool = True,
                 shuffle_data: bool = True,
                 print_output: bool = False,
                 log_level: int = log_levels.WARNING,
                 print_level: int = print_levels.NONE):
        """Constructor of the Classification class.

        Takes raw_data as necessary parameter and some more optional parameters, which are specified below.
        Stores the following values as protected attributes:
        + self._original_data: Original data.
        + self._scaled_data: Scaled original data.
        + self._omitted_data: During initialization and testing omitted classless data.
        + self._learning_data: In _initialize() assigned learning data.
        + self._testing_data: In _initialize() assigned testing data, all new testing data is stored here as well.
        + self._data_range: The original range of all data samples in each dimension.
        + self._scale_factor: The scaling factor for each dimension, with which all samples in this object were scaled.
        + self._calculated_classes_testset: All calculated classes for samples in self._testing_data in the same order (useful for printing).
        + self._densities_testset: All estimated densities for the samples in the testing data set.
        + self._classificators: List of all in _perform_classification() computed classificators. One for each class.
        + self._de_objects: List of all in in _perform_classification() computed DensityEstimation objects (useful for plotting).
        + self._performed_classification: Check, if classfication was already performed for this object.
        + self._time_used: Time used for the learning process.

        :param raw_data: DataSet on which to perform learning (and if 0 < percentage < 1 also testing).
        :param data_range: Optional. If the user knows the original range of the dataset, they can specify it here.
        :param split_percentage: Optional. If a percentage of raw data should be used as testing data: 0 < percentage < 1. Default 1.0.
        :param split_evenly: Optional. Only relevant when 0 < percentage < 1. Conditional parameter, which indicates whether the learning data sets
        for each class should be of near equal size. Default True.
        :param shuffle_data: Optional. Indicates, whether the data should be randomly shuffled in the initialization step. Default True.
        :param print_output: Optional. Print log statements to console
        :param log_level: Optional. Set the log level for this instance. Only statements of the given level or higher will be written to the log file
        :param print_level: Optional.  Set the level for print statements. Only statements of the given level or higher will be written to the console
        """
        self._original_data = raw_data
        self._scaled_data = None
        self._omitted_data = None
        self._learning_data = None
        self._testing_data = None
        self._data_range = data_range
        self._scale_factor = None
        self._calculated_classes_testset = np.array([])
        self._densities_testset = []
        self._classificators = []
        self._de_objects = []
        self._performed_classification = False
        self._time_used = None
        self._print_output = print_output
        self.log_util = LogUtility(log_level=log_level, print_level=print_level)
        # for compatibility with old code
        if print_output is True and print_level == print_levels.NONE:
            self.log_util.set_print_level(print_levels.INFO)
        self.log_util.set_print_prefix('Classification')
        self.log_util.set_log_prefix('Classification')
        self._initialize((split_percentage if isinstance(split_percentage, float) and (1 > split_percentage > 0) else 1.0), split_evenly,
                         shuffle_data)

    def __call__(self, data_to_evaluate: 'DataSet', print_removed: bool = True) -> 'DataSet':
        """Evaluate classes for samples in input data and create a new DataSet from those same samples and classes.

        When this method is called, classification needs to already be performed. If it is not, an AttributeError is raised.
        Also the input data mustn't be empty. If it is, a ValueError is raised.
        Before classification, input data is scaled to match the scaling of learning data of this object; samples out of bounds after scaling are
        removed and by default printed to stdout.

        :param data_to_evaluate: Data whose samples are to be classified.
        :param print_removed: Optional. Conditional parameter, which specifies whether during scaling removed samples should be printed.
        :return: New DataSet, which consists of samples from input DataSet and for those samples computed classes.
        """
        if not self._performed_classification:
            raise AttributeError("Classification needs to be performed on this object first.")
        if data_to_evaluate.is_empty():
            raise ValueError("Can't classificate empty dataset.")
        self.log_util.log_info(
            "---------------------------------------------------------------------------------------------------------------------------------")
        self.log_util.log_info("Evaluating classes of %s DataSet..." % data_to_evaluate.get_name())
        evaluate = self._internal_scaling(data_to_evaluate, print_removed=print_removed)
        if evaluate.is_empty():
            raise ValueError("All given samples for classification were out of bounds. Please only evaluate classes for samples in unscaled range: "
                             "\n[%s]\n[%s]\nwith this classification object" %
                             (', '.join([str(x) for x in self._data_range[0]]), ', '.join([str(x) for x in self._data_range[1]])))
        evaluated_data = DataSet(tuple([evaluate[0], np.array(self._classificate(evaluate))]), name="%s_evaluated_classes" %
                                                                                                    data_to_evaluate.get_name(), label="class")
        del self._densities_testset[(len(self._densities_testset) - data_to_evaluate.get_length()):]
        return evaluated_data

    def test_data(self, new_testing_data: DataSet,
                  print_output: bool = True,
                  print_removed: bool = True,
                  print_incorrect_points: bool = False) -> dict:
        """Test new data with the classificators of a Classification object.

        As most of other public methods of Classification, classification already has to be performed before this method is called. Otherwise an
        AttributeError is raised.
        In case the input testing data is empty, a ValueError is raised.
        Test data is scaled with the same factors as the scaled original data (self._scaled_data) and samples out of bounds after scaling are removed.
        Only test data samples with known classes can be used for testing; the omitted rest is stored into self._omitted_data.
        Test data with known classes and samples only inside of bounds is stored into self._testing_data, results are calculated and printed
        (default) if the user specified it.

        :param new_testing_data: Test DataSet for which classificators should be tested.
        :param print_output: Optional. Conditional parameter, which specifies whether results of testing should be printed. Default True.
        :param print_removed: Optional. Conditional parameter, which specifies whether during scaling removed samples should be printed. Default True.
        :param print_incorrect_points: Conditional parameter, which specifies whether the incorrectly mapped points should be printed. Default False.
        :return: DataSet, which contains all classless samples that were omitted.
        """
        if not self._performed_classification:
            raise AttributeError("Classification needs to be performed on this object first.")
        if new_testing_data.is_empty():
            raise ValueError("Can't test empty dataset.")
        self.log_util.log_info(
            "---------------------------------------------------------------------------------------------------------------------------------")
        self.log_util.log_info("Testing classes of %s DataSet..." % new_testing_data.get_name())
        new_testing_data.set_label("class")
        evaluate = self._internal_scaling(new_testing_data, print_removed=print_removed)
        if evaluate.is_empty():
            raise ValueError("All given samples for testing were out of bounds. Please only test samples in unscaled range: "
                             "\n[%s]\n[%s]\nwith this classification object" %
                             (', '.join([str(x) for x in self._data_range[0]]), ', '.join([str(x) for x in self._data_range[1]])))
        omitted_data, used_data = evaluate.split_without_labels()
        if not omitted_data.is_empty():
            self.log_util.log_info("Omitted some classless samples during testing and added them to omitted sample collection of this object.")
        self._omitted_data.concatenate(omitted_data)
        self._scaled_data.concatenate(used_data)
        self._testing_data.concatenate(used_data)
        calculated_new_testclasses = self._classificate(used_data)
        self._calculated_classes_testset = np.concatenate((self._calculated_classes_testset, calculated_new_testclasses))
        if print_output:
            self._print_evaluation(used_data, calculated_new_testclasses,
                                   self._densities_testset[(len(self._densities_testset) - new_testing_data.get_length()):],
                                   print_incorrect_points)
        return self._evaluate(used_data, calculated_new_testclasses)

    def get_original_data(self) -> 'DataSet':
        ret_val = self._original_data.copy()
        ret_val.set_name(self._original_data.get_name())
        return ret_val

    def get_omitted_data(self) -> 'DataSet':
        ret_val = self._omitted_data.copy()
        ret_val.set_name(self._omitted_data.get_name())
        return ret_val

    def get_learning_data(self) -> 'DataSet':
        ret_val = self._learning_data.copy()
        ret_val.set_name(self._learning_data.get_name())
        return ret_val

    def get_testing_data(self) -> 'DataSet':
        ret_val = self._testing_data.copy()
        ret_val.set_name(self._testing_data.get_name())
        return ret_val

    def get_dataset_range(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._data_range

    def get_scale_factor(self) -> float:
        return self._scale_factor

    def get_calculated_classes_testset(self) -> List[int]:
        return self._calculated_classes_testset.copy()

    def get_time_used(self) -> float:
        return self._time_used

    def get_density_estimation_results(self) -> Tuple[List[StandardCombi], List[DensityEstimation]]:
        return self._classificators, self._de_objects

    def _initialize(self, percentage: float, split_evenly: bool, shuffle_data: bool) -> None:
        """Initialize data for performing classification.

        Calculates, which parts of the original data should be used as learning and testing data.
        If percentage is 1, all of the original data is used as learning data.
        Any classless samples in the original dataset are removed and stored in self._omitted_data first.
        Scaling to range (0.005, 0.995) is performed either simply based on boundary samples (default) or by the original data range (if
        specified by the user in the constructor). If the latter, samples out of bounds are removed, printed to stdout and the user is notified.
        Before splitting the data, it is shuffled (if corresponding constructor parameter is set) to ensure random splitting and the boundary samples
        are moved to the front to guarantee that they are in the learning dataset.

        :param percentage: Percentage of original data, which to use as learning dataset.
        :param split_evenly: Only relevant when 0 < percentage < 1. Conditional parameter, which indicates whether the learning data sets
        for each class should be of near equal size.
        :param shuffle_data: Conditional parameter, which indicates whether the learning set should be shuffled before initialization.
        :return: None
        """
        self._scaled_data = self._original_data.copy()
        self._scaled_data.set_name(self._original_data.get_name())
        self._scaled_data.set_label("class")
        self._omitted_data, used_data = self._scaled_data.split_without_labels()
        self._omitted_data.set_name("%s_omitted" % self._scaled_data.get_name())
        used_data.set_name(self._scaled_data.get_name())
        self._scaled_data = used_data
        if self._scaled_data.is_empty():
            raise ValueError("Can't perform classification learning on empty or classless DataSet.")
        if self._data_range is not None:
            if any(x <= y for x, y in zip(self._data_range[1], self._data_range[0])):
                raise ValueError("Invalid dataset range.")
            self._scale_factor = 0.99 / (self._data_range[1] - self._data_range[0])
            self._scaled_data = self._internal_scaling(self._scaled_data, print_removed=True)
        else:
            self._scaled_data.scale_range((0.005, 0.995), override_scaling=True)
            self._data_range = (self._scaled_data.get_original_min(), self._scaled_data.get_original_max())
            self._scale_factor = self._scaled_data.get_scaling_factor()
        if not self._omitted_data.is_empty():
            self.log_util.log_info("Omitted some classless samples during initialization and added them to omitted sample collection of this object.")
            self._omitted_data.shift_value(-self._data_range[0], override_scaling=True)
            self._omitted_data.scale_factor(self._scale_factor, override_scaling=True)
            self._omitted_data.shift_value(0.005, override_scaling=True)
        if shuffle_data:
            self._scaled_data.shuffle()
        self._scaled_data.move_boundaries_to_front()
        if split_evenly:
            data_classes = self._scaled_data.split_labels()
            data_learn_list = [x.split_pieces(percentage)[0] for x in data_classes]
            data_test_list = [x.split_pieces(percentage)[1] for x in data_classes]
            data_learn = DataSet.list_concatenate(data_learn_list)
            data_test = DataSet.list_concatenate(data_test_list)
        else:
            data_learn, data_test = self._scaled_data.split_pieces(percentage)
        self._learning_data = data_learn
        self._learning_data.set_name("%s_learning_data" % self._scaled_data.get_name())
        self._testing_data = data_test
        self._testing_data.set_name("%s_testing_data" % self._scaled_data.get_name())

    def _classificate(self, data_to_classificate: DataSet) -> np.ndarray:
        """Calculate classes for samples of input data.

        Computes the densities of each class for every sample. The class, which corresponds to the highest density for a sample is assigned to it.
        Classes are stored into a list in the same order as the corresponding samples occur in the input data.

        :param data_to_classificate: DataSet whose samples are to be classified.
        :return: List of computed classes in the same order as their corresponding samples.
        """
        density_data = list(zip(*[x(data_to_classificate[0]) for x in self._classificators]))
        self._densities_testset += density_data
        return np.argmax(density_data, axis=1).flatten()

    def _internal_scaling(self, data_to_check: DataSet, print_removed: bool = False) -> 'DataSet':
        """Scale data with the same factors as the original data was scaled (self._scaled_data).

        If the input data to check is already scaled and its scaling doesn't match that of the original scaled data, a ValueError is raised.
        If not already scaled, the input data to check will be scaled with the same factors the original data was.
        Any samples out of bounds after scaling are removed, printed to stdout if print_removed is True and a the user is notified.

        :param data_to_check: DataSet, which needs to be checked for scaling and scaled if necessary.
        :param print_removed: Optional. Conditional parameter, which indicates whether any during scaling removed samples should be printed.
        :return: Scaled input dataset without samples out of bounds.
        """
        if data_to_check.is_scaled():
            if not self._scaled_data.same_scaling(data_to_check):
                raise ValueError("Provided DataSet's scaling doesn't match the internal scaling of Classification object.")
        else:
            data_to_check.shift_value(-self._data_range[0], override_scaling=False)
            data_to_check.scale_factor(self._scale_factor, override_scaling=False)
            data_to_check.shift_value(0.005, override_scaling=False)
        remove_indices = [i for i, x in enumerate(data_to_check[0]) if any([(y < 0.0049) for y in x]) or any([(y > 0.9951) for y in x])]
        removed_samples = data_to_check.remove_samples(remove_indices)
        if not removed_samples.is_empty():
            self.log_util.log_info("During internal scale checking of %s DataSet some samples were removed due to them being out of bounds of "
                                   "classificators." % data_to_check.get_name())
            if print_removed:
                self.log_util.log_info("Points removed during scale checking:")
                for i, x in enumerate(removed_samples[0]):
                    self.log_util.log_info("{0} : {1} | class {2}".format(i, x, removed_samples[1][i]))
        return data_to_check

    @staticmethod
    def _evaluate(testing_data: DataSet, calculated_classes: np.ndarray) -> dict:
        """Directly evaluate results of tested set.

        :param testing_data: nput testing data for which to print the results.
        :param calculated_classes: Input calculated classes for specified testing data.
        :return: Dictionary of all results.
        """
        if testing_data.get_length() != len(calculated_classes):
            raise ValueError("Samples of testing DataSet and its calculated classes have to be the same amount.")
        number_wrong = sum([0 if (x == y) else 1 for x, y in zip(testing_data[1], calculated_classes)])
        return {"Wrong mappings": number_wrong,
                "Total mappings": len(calculated_classes),
                "Percentage correct": 1.0 - (number_wrong / len(calculated_classes)),
                "Percentage correct (str)": ("%2.2f%%" % ((1.0 - (number_wrong / len(calculated_classes))) * 100))}

    @staticmethod
    def _print_evaluation(testing_data: DataSet,
                          calculated_classes: np.ndarray,
                          density_testdata: List[np.ndarray],
                          print_incorrect_points: bool = True) -> None:
        """Print the results of some specified testing data to stdout.

        Only prints the evaluation, if the input is valid.
        Prints the number and percentage of incorrectly computed samples.
        Prints all samples of input test data that were classified incorrectly, if the corresponding parameter is set.

        :param testing_data: Input testing data for which to print the results.
        :param calculated_classes: Input calculated classes for specified testing data.
        :param density_testdata: List of densities for the corresponding testing data.
        :param print_incorrect_points: Optional. Conditional parameter, which indicates whether the incorrectly mapped points should be printed to
        stdout.
        :return: None
        """
        if testing_data.is_empty():
            warnings.formatwarning = lambda msg, ctg, fname, lineno, file=None, line=None: "%s:%s: %s: %s\n" % (fname, lineno, ctg.__name__, msg)
            warnings.warn("Nothing to print; input test dataset is empty.", stacklevel=3)
            return
        if testing_data.get_length() != len(calculated_classes):
            raise ValueError("Samples of testing DataSet and its calculated classes have to be the same amount.")
        number_wrong = sum([0 if (x == y) else 1 for x, y in zip(testing_data[1], calculated_classes)])
        indices_wrong = [i for i, c in enumerate(testing_data[1]) if c != calculated_classes[i]]
        logUtil.log_info("Number of wrong mappings: {0} ".format(number_wrong))
        logUtil.log_info("Number of total mappings: {0}".format(len(calculated_classes)))
        percentage_wrong = "%2.2f%%" % ((1.0 - (number_wrong / len(calculated_classes))) * 100)
        logUtil.log_info("Percentage of correct mappings: {0}".format(percentage_wrong))
        if number_wrong != 0 and print_incorrect_points:
            logUtil.log_info(
                "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
            logUtil.log_info("Points mapped incorrectly:")
            points = ''
            for i, wr in enumerate(indices_wrong):
                points += "{0}: {1} | correct class: {2}, calculated class: {3} | ".format(i, testing_data[0][wr],
                                                                                           testing_data[1][wr],
                                                                                           calculated_classes[wr])
                d_c = ""
                for j, x in enumerate(density_testdata[wr]):
                    d_c += "density_class{0}: {1}, ".format(j, x)
                points += d_c
            logUtil.log_info("Points mapped incorrectly: {0}".format(points))

    def _process_performed_classification(self,
                                          operation_list: List[Tuple[StandardCombi, DensityEstimation]],
                                          start_time: float,
                                          print_metrics: bool) -> None:
        """Perform the last core-steps of the classification process.

        Prints the time used for performing classification.
        Already evaluates testing data, if the original data was split into learning and testing data in _initialize().

        :param operation_list: List of Tuples of StandardCombi and DensityEstimation objects, which each can be assigned to a class.
        :param start_time: Time when the performing of classification of this object started.
        :param print_metrics: Optional. Conditional parameter, which indicates whether time metrics should be printed immediately after completing
        the learning process.
        :return: None
        """
        self._classificators = [x[0] for x in operation_list]
        self._de_objects = [x[1] for x in operation_list]
        self._performed_classification = True
        self._time_used = time.time() - start_time
        if print_metrics:
            self.log_util.log_info()
            self.log_util.log_info(
                "=================================================================================================================================")
        self.log_util.log_info("Performed Classification of '%s' DataSet." % self._scaled_data.get_name())
        if print_metrics:
            self.log_util.log_info("Time used: %.10f seconds" % self._time_used)
        if not self._testing_data.is_empty():
            start_time = time.time()
            self._calculated_classes_testset = self._classificate(self._testing_data)
            self._time_used += time.time() - start_time

    def perform_classification(self,
                               masslumping: bool = True,
                               lambd: float = 0.0,
                               minimum_level: int = 1,
                               maximum_level: int = 5,
                               reuse_old_values: bool = False,
                               one_vs_others: bool = False,
                               pre_scaled_data: bool = False,
                               print_metrics: bool = True) -> None:
        """Create GridOperation and DensityEstimation objects for each class of all samples and store them into lists.

        This method is only called once.
        First the learning dataset is split into its classes in separate DataSets and then the DataSet.density_estimation() function is called for
        each of the single-class-DataSets.
        The DensityEstimation objects are mainly used for plotting the combination-scheme later.

        :param masslumping: Optional. Conditional Parameter, which indicates whether masslumping should be enabled for DensityEstimation.
        :param lambd: Optional. Parameter, which adjusts the 'smoothness' of DensityEstimation results.
        :param minimum_level: Optional. Minimum Level of Sparse Grids on which to perform DensityEstimation.
        :param maximum_level: Optional. Maximum Level of Sparse Grids on which to perform DensityEstimation.
        :param one_vs_others: Optional. Data from other classes will be included with a weighted label < 0.
        :param reuse_old_values: Optional. R-values and b-values will be re-used across refinements and component grids.
        :param pre_scaled_data: Optional. Deactivates data scaling in the grid operation.
        :param print_metrics: Optional. Conditional parameter, which indicates whether time metrics should be printed immediately after completing.
        the learning process.
        :return: None
        """
        if self._performed_classification:
            raise ValueError("Can't perform classification for the same object twice.")
        start_time = time.time()
        if one_vs_others:
            learning_data_classes = self._learning_data.split_one_vs_others()
        else:
            learning_data_classes = self._learning_data.split_labels()
        operation_list = [x.density_estimation(masslumping=masslumping, lambd=lambd,
                                               minimum_level=minimum_level, maximum_level=maximum_level,
                                               one_vs_others=one_vs_others, reuse_old_values=reuse_old_values,
                                               pre_scaled_data=pre_scaled_data,
                                               plot_de_dataset=False, plot_density_estimation=False,
                                               plot_combi_scheme=False, plot_sparsegrid=False)
                          for x in learning_data_classes]
        self._process_performed_classification(operation_list, start_time, print_metrics)

    def perform_classification_dimension_wise(self,
                                              masslumping: bool = True,
                                              lambd: float = 0.0,
                                              minimum_level: int = 1,
                                              maximum_level: int = 5,
                                              reuse_old_values: bool = False,
                                              numeric_calculation: bool = True,
                                              margin: float = 0.5,
                                              tolerance: float = 0.01,
                                              max_evaluations: int = 256,
                                              rebalancing: bool = False,
                                              modified_basis: bool = False,
                                              boundary: bool = False,
                                              one_vs_others: bool = False,
                                              error_calculator: ErrorCalculator = None,
                                              pre_scaled_data: bool = False,
                                              print_metrics: bool = True) -> None:
        """Create dimension-wise GridOperation and DensityEstimation objects for each class of all samples and store them into lists.

        This method is only called once.
        First the learning dataset is split into its classes in separate DataSets and then the DataSet.density_estimation() function is called for
        each of the single-class-DataSets.
        The DensityEstimation objects are mainly used for plotting the combination-scheme later.

        :param masslumping: Optional. Conditional Parameter which indicates whether masslumping should be enabled for DensityEstimation
        :param lambd: Optional. Parameter which adjusts the 'smoothness' of DensityEstimation results
        :param minimum_level: Optional. Minimum Level of Sparse Grids on which to perform DensityEstimation
        :param maximum_level: Optional. Maximum Level of Sparse Grids on which to perform DensityEstimation
        :param reuse_old_values: Optional. R-values and b-values will be re-used across refinements and component grids.
        :param numeric_calculation: Optional. Use numerical calculation for the integral.
        :param margin: Optional.
        :param tolerance: Optional. Error tolerance. Refinement stops if this threshold is reached
        :param max_evaluations: Optional. Maximum number of points. The refinement will stop when it exceeds this limit.
        :param rebalancing: Optional. Activate rebalancing of refinement trees.
        :param modified_basis: Optional. Use the modified basis function..
        :param boundary: Optional. Put points on the boundary.
        :param one_vs_others: Optional. Data from other classes will be included with a weighted label < 0.
        :param error_calculator: Optional. Explicitly pass the error calculator that you want to use.
        :param pre_scaled_data: Optional. Data will not be scaled within the density estimation operation
        :param print_metrics: Optional.
        :return: None
        """
        if self._performed_classification:
            raise ValueError("Can't perform classification for the same object twice.")
        start_time = time.time()
        if one_vs_others:
            learning_data_classes = self._learning_data.split_one_vs_others()
        else:
            learning_data_classes = self._learning_data.split_labels()
        operation_list = [x.density_estimation_dimension_wise(masslumping=masslumping,
                                                              lambd=lambd,
                                                              minimum_level=minimum_level,
                                                              maximum_level=maximum_level,
                                                              reuse_old_values=reuse_old_values,
                                                              numeric_calculation=numeric_calculation,
                                                              margin=margin,
                                                              tolerance=tolerance,
                                                              max_evaluations=max_evaluations,
                                                              rebalancing=rebalancing,
                                                              modified_basis=modified_basis,
                                                              boundary=boundary,
                                                              one_vs_others=one_vs_others,
                                                              error_calculator=error_calculator,
                                                              pre_scaled_data=pre_scaled_data,
                                                              plot_de_dataset=False,
                                                              plot_density_estimation=False,
                                                              plot_combi_scheme=False,
                                                              plot_sparsegrid=False) for x in learning_data_classes]
        self._process_performed_classification(operation_list, start_time, print_metrics)

    def continue_dimension_wise_refinement(self,
                                           tolerance: float = 0.01,
                                           max_time: float = None,
                                           max_evaluations: int = 256,
                                           min_evaluations: int = 1):
        """Create dimension-wise GridOperation and DensityEstimation objects for each class of all samples and store them into lists.

        This method requires 'perform_classification_dimension_wise' to be called before.
        This method continues the refinement started by 'perform_classification_dimension_wise'. It allows refinement
        to be performed iteratively, so that performance evaluations can be performed between each refinement iteration.

        ----------
        :param tolerance: Optional. Error tolerance. Refinement stops if this threshold is reached
        :param max_time: Optional. Maximum compute time. The refinement will stop when it exceeds this time.
        :param max_evaluations: Optional. Maximum number of points. The refinement will stop when it exceeds this limit.
        :param min_evaluations: Optional. Minimum number of points. The refinement will not stop until it exceeds this limit.
        """
        if self._classificators is not None and self._de_objects is not None:
            for combi_object in self._classificators:
                combi_object.continue_adaptive_refinement(tol=tolerance,
                                                          max_time=max_time,
                                                          max_evaluations=max_evaluations,
                                                          min_evaluations=min_evaluations)
            self.log_util.log_info("Continued Classification of '{0}' DataSet.".format(self._learning_data.get_name()))
            self.log_util.log_info(
                "_________________________________________________________________________________________________________________________________")
            self.log_util.log_info(
                "---------------------------------------------------------------------------------------------------------------------------------")

    def evaluate(self) -> dict:
        """Evaluate results of all testing data stored within this object.

        As most of other public methods of Classification, classification already has to be performed before this method is called. Otherwise an
        AttributeError is raised.
        In case self._testing_data is empty, a warning is issued and the method returns without printing anything.

        :return: Dictionary of all results.
        """
        if not self._performed_classification:
            raise AttributeError("Classification needs to be performed on this object first.")
        if self._testing_data.is_empty():
            raise ValueError("Nothing to evaluate; test dataset of this object is empty.")
        if self._testing_data.get_length() != len(self._calculated_classes_testset):
            raise ValueError("Samples of testing DataSet and its calculated classes have to be the same amount.")
        number_wrong = sum([0 if (x == y) else 1 for x, y in zip(self._testing_data[1], self._calculated_classes_testset)])
        return {"Time used": self._time_used,
                "Wrong mappings": number_wrong,
                "Total mappings": len(self._calculated_classes_testset),
                "Percentage correct": 1.0 - (number_wrong / len(self._calculated_classes_testset)),
                "Percentage correct (str)": ("%2.2f%%" % ((1.0 - (number_wrong / len(self._calculated_classes_testset))) * 100))}

    def print_evaluation(self, print_incorrect_points: bool = True) -> None:
        """Print results of all testing data that was evaluated with this object.

        As most of other public methods of Classification, classification already has to be performed before this method is called. Otherwise an
        AttributeError is raised.
        In case self._testing_data is empty, a warning is issued and the method returns without printing anything.

        :return: None
        """
        if not self._performed_classification:
            raise AttributeError("Classification needs to be performed on this object first.")
        if self._testing_data.is_empty():
            warnings.formatwarning = lambda msg, ctg, fname, lineno, file=None, line=None: "%s:%s: %s: %s\n" % (fname, lineno, ctg.__name__, msg)
            warnings.warn("Nothing to print; test dataset of this object is empty.", stacklevel=3)
            return
        self.log_util.log_info(
            "---------------------------------------------------------------------------------------------------------------------------------")
        self.log_util.log_info("Printing evaluation of all current testing data...")
        self._print_evaluation(self._testing_data, np.array(self._calculated_classes_testset), self._densities_testset, print_incorrect_points)

    def plot(self,
             plot_class_dataset: bool = False,
             plot_class_density_estimation: bool = False,
             plot_class_combi_scheme: bool = False,
             plot_class_sparsegrid: bool = False,
             file_path: str = None) -> None:
        """Plot a Classification object.

        As most of other public methods of Classification, classification already has to be performed before this method is called. Otherwise an
        AttributeError is raised.
        The user can specify exactly what to plot with conditional parameters of this method.

        :param plot_class_dataset: Optional. Conditional parameter, which specifies whether the learning DataSet should be plotted. Default False.
        :param plot_class_density_estimation: Optional. Conditional parameter, which specifies whether the density estimation of each class should be
        plotted. Default False.
        :param plot_class_combi_scheme: Optional. Conditional parameter, which specifies whether the resulting combi schemes of each class should be
        plotted. Default False.
        :param plot_class_sparsegrid: Optional. Conditional parameter, which specifies whether the resulting sparsegrids of each class should be
        plotted. Default False.
        :param file_path: Optional. file path that points to where the plots should be saved
        :return: None
        """
        if not self._performed_classification:
            raise AttributeError("Classification needs to be performed on this object first.")
        if plot_class_dataset:
            self._learning_data.plot()
        if plot_class_density_estimation:
            filename = file_path + '_contour' if file_path is not None else None
            for x in self._classificators:
                if filename is not None:
                    while os.path.isfile(filename + '.png'):
                        filename = filename + '+'
                x.plot(contour=True, filename=filename)
        if plot_class_combi_scheme:
            filename = file_path + '_combi_scheme' if file_path is not None else None
            for x, y in zip(self._classificators, self._de_objects):
                if filename is not None:
                    while os.path.isfile(filename + '.png'):
                        filename = filename + '+'
                x.print_resulting_combi_scheme(operation=y, filename=filename)
        if plot_class_sparsegrid:
            filename = file_path + '_sparsegrid' if file_path is not None else None
            for x in self._classificators:
                if filename is not None:
                    while os.path.isfile(filename + '.png'):
                        filename = filename + '+'
                x.print_resulting_sparsegrid(markersize=20, filename=filename)


class Clustering:
    """Type of objects, that cluster data based on some previously performed learning.
    """

    def __init__(self, raw_data: 'DataSet', number_nearest_neighbors: int = 5, edge_cutting_threshold: float = 0.25,
                 log_level: int = log_levels.WARNING, print_level: int = print_levels.NONE):
        """Constructor of the Clustering class.

        Takes raw_data as necessary parameter and some more optional parameters, which are specified below.
        Stores the following values as protected attributes:
        + self._original_data: Original data.
        + self._scaled_data: Scaled original data.
        + self._clustered_data: Original data with computed labels.
        + self._label: Name-type of the assigned labels.
        + self._number_nn: Number of nearest neighbors for the connected graph.
        + self._threshold: Edge cutting threshold.
        + self._clusterinator: The in _perform_clustering() computed density function for the scaled data.
        + self._de_objects: The in _perform_clustering() computed DensityEstimation object (useful for plotting).
        + self._density_range: The range of minimal and maximal possible densities in the density function.
        + self._all_edges: All edges in the initial nearest neighbor graph.
        + self._remaining_edges: AAll edges of the cut nearest neighbor graph after the threshold is applied. With noise samples.
        + self._connected_components: List of ndarrays of samples, which form the connected components computed in _compute_connected_components().
        + self._connected_edges: All edges of the cut nearest neighbor graph after the threshold is applied. Without noise samples.
        + self._connected_samples: All samples within _connected_edges.
        + self._noise_edges: All edges of detected noise samples after the threshold is applied.
        + self._noise_samples: All samples within _noise_edges.
        + self._performed_clustering: Check, if clustering was already performed for this object.
        + self._time_used: Time used for the learning process.

        :param raw_data: DataSet on which to perform learning (and if 0 < percentage < 1 also testing).
        :param number_nearest_neighbors: Optional. Specifies the number of neighbors for the connected graph. Default 5.
        :param edge_cutting_threshold: Optional. Specifies the edge cutting threshold, which later assists in omitting some edges. Default 0.25.
        :param log_level: Optional. Set the log level for this instance. Only statements of the given level or higher will be written to the log file
        :param print_level: Optional.  Set the level for print statements. Only statements of the given level or higher will be written to the console
        """
        self._original_data = raw_data
        self._scaled_data = None
        self._clustered_data = None
        self._label = 'cluster'
        self._number_nn = number_nearest_neighbors + 1
        self._threshold = edge_cutting_threshold
        self._clustertinator = None
        self._de_object = None
        self._density_range = None
        self._all_edges = None
        self._remaining_edges = None
        self._connected_components = []
        self._connected_edges = []
        self._connected_samples = None
        self._noise_edges = []
        self._noise_samples = None
        self._performed_clustering = False
        self._time_used = None
        self.log_util = LogUtility(log_level=log_level, print_level=print_level)
        self.log_util.set_print_prefix('Clustering')
        self.log_util.set_log_prefix('Clustering')
        self._initialize()

    def get_original_data(self) -> DataSet:
        ret_val = self._original_data.copy()
        ret_val.set_name(self._original_data.get_name())
        return ret_val

    def get_clustered_data(self) -> DataSet:
        ret_val = self._clustered_data.copy()
        ret_val.set_name(self._clustered_data.get_name())
        return ret_val

    def get_connected_samples_dataset(self) -> DataSet:
        ret_val = self._connected_samples.copy()
        ret_val.set_name(self._connected_samples.get_name())
        return ret_val

    def get_noise_samples_dataset(self) -> DataSet:
        ret_val = self._noise_samples.copy()
        ret_val.set_name(self._noise_samples.get_name())
        return ret_val

    def get_max_number_nearest_neighbors(self) -> int:
        return self._number_nn

    def get_edge_cutting_threshold(self) -> float:
        return self._threshold

    def get_time_used(self) -> float:
        return self._time_used

    def get_scaled_data(self) -> DataSet:
        ret_val = self._scaled_data.copy()
        ret_val.set_name(self._scaled_data.get_name())
        return ret_val

    def _initialize(self) -> None:
        """Initialize data for performing clustering.

        The original data is scaled to range (0.005, 0.995) for the learning process occurring later.

        :return: None
        """
        if self._original_data.is_empty():
            raise ValueError("Can't perform clustering on empty DataSet.")
        self._scaled_data = self._original_data.copy()
        self._scaled_data.set_name(self._original_data.get_name())
        self._scaled_data.set_label(self._label)
        self._scaled_data.scale_range((0.005, 0.995), override_scaling=True)

    def _compute_nearest_neighbors_connected(self) -> None:
        """Build the initial nearest neighbor graph without noise.

        The sklearn.neighbors.Neighbors functionality is used to fit the scaled data onto itself and detect all nearest neighbors.
        All this way computed edges are ridded of redundancy and then all edges with a medium density below the threshold are omitted.
        This results in some samples without any connecting edges (noise), whose will be handled in _compute_nearest_neighbors_noise().

        :return: None
        """
        # search for nearest neighbors and store edges which weren't cut
        neigh = neighbors.NearestNeighbors(n_neighbors=self._number_nn)
        neigh.fit(self._scaled_data[0])
        self._all_edges = list(set(tuple(sorted(x)) for x in [[x[0], y] for x in neigh.kneighbors(self._scaled_data[0], return_distance=False)
                                                              for y in x[1:]]))

        centralvalues_edges = np.array([np.array([(self._scaled_data[0][e[0]] + self._scaled_data[0][e[1]]) / 2])
                                        for e in self._all_edges]).reshape((len(self._all_edges), self._scaled_data.get_dim()))
        centralvalues_densities = self._clustertinator(centralvalues_edges)
        centralvalues_dens_percentage = np.array(centralvalues_densities / self._density_range[1])
        centralvalues_dens_percentage[centralvalues_dens_percentage > 1] = 1.0
        centralvalues_dens_percentage[centralvalues_dens_percentage < 0] = 0.0
        self._connected_edges = [e for i, e in enumerate(self._all_edges) if (centralvalues_dens_percentage[i] > self._threshold)]

    def _compute_nearest_neighbors_noise(self) -> None:
        """Detect all noise samples within the initial nearest neighbor graph.

        The noise samples created in _compute_nearest_neighbors_connected() are detected and for each of them exactly one edge to the nearest
        connected sample is created. Those edges are then added to the ones previously computed.

        :return: None
        """
        # filter single samples
        singles_indices = [i for i in np.arange(self._scaled_data.get_length()) if i not in np.unique(self._connected_edges)]
        self._connected_samples = self._scaled_data.copy()
        self._noise_samples = self._connected_samples.remove_samples(singles_indices)

        if self._noise_samples.is_empty():
            self._noise_edges = []
            self._remaining_edges = np.array((sorted(self._connected_edges)))
        elif not self._connected_samples.is_empty():
            cluster_neigh = neighbors.NearestNeighbors(n_neighbors=1)
            cluster_neigh.fit(self._connected_samples[0])
            singles_neigh = cluster_neigh.kneighbors(self._noise_samples[0], return_distance=False)
            self._noise_edges = [(singles_indices[i], np.where(self._scaled_data[0] == self._connected_samples[0][x[0]])[0][0])
                                 for i, x in enumerate(singles_neigh)]
            self._remaining_edges = np.array(sorted(self._noise_edges + self._connected_edges))

    def _compute_connected_components(self) -> None:
        """Detect all connected components or clusters.

        If only noise samples were detected while building the nearest neighbor graph, every sample is its own connected component. Else recursive
        depth-first-search is used to detect the connected components.

        :return: None
        """
        noisy = self._connected_samples.is_empty()
        self._connected_components = [np.array([x]) for x in range(self._scaled_data.get_length())] if noisy else self._depth_first_search()

    def _label_samples(self) -> None:
        """Assign each sample of the original data a label based on detected clusters.

        :return: None
        """
        clusters = np.empty((self._scaled_data.get_length(),), dtype=np.int64)
        for i, all_indices_component in enumerate(self._connected_components):
            clusters[all_indices_component] = i
        self._clustered_data = DataSet((self._original_data[0], clusters), name="Clustered_%s" % self._original_data.get_name(), label=self._label)

    def _depth_first_search(self) -> List[np.ndarray]:
        """Depth-first-search for detecting the connected components.

        For each connected component, recursive depth-first-search is used to detect all samples within this component.

        :return: List of ndarrays of samples (List of connected components).
        """
        # depth first search
        all_nodes = np.arange(self._scaled_data.get_length())
        visited_nodes = np.full((self._scaled_data.get_length(),), False)
        connected_components = []
        while not all(visited_nodes):
            unvisited_nodes = all_nodes[np.invert(visited_nodes)]
            # sorting not necessary but nice, can be removed. but always look on the bright side of sorting
            current_snake = np.sort(self._dfs_inner_recursive(unvisited_nodes[0], visited_nodes, self._remaining_edges, np.arange(0)))
            connected_components += [current_snake]
        return connected_components

    def _dfs_inner_recursive(self,
                             current_node: int,
                             visited_nodes: np.ndarray,
                             edges: np.ndarray,
                             connected_component: np.ndarray) -> List[np.ndarray]:
        """Inner helper-function for recursive depth-first-search.

        After every recursive step of this function, one connected component is returned.

        :param current_node: Active sample.
        :param visited_nodes: All already visited samples.
        :param edges: All possible edges from the active sample.
        :param connected_component: Current connected component.
        :return: Completed current connected component.
        """
        connected_component = np.concatenate((connected_component, np.array([current_node])))
        visited_nodes[current_node] = True
        current_edges = edges[np.where(edges == current_node)[0]]
        for x in current_edges:
            next_node = x[1] if x[0] == current_node else x[0]
            if visited_nodes[next_node]:
                continue
            connected_component = self._dfs_inner_recursive(next_node, visited_nodes, edges, connected_component)
        return connected_component

    def _process_performed_clustering(self, start_time: float, print_metrics: bool) -> None:
        """Perform the last core-steps of the clustering process.

        Prints the time used for performing clustering.
        Performs all important steps of building and cutting the nearest neighbor graph and detecting the connected components.

        :param start_time: Time when the performing of classification of this object started.
        :param print_metrics: Optional. Conditional parameter, which indicates whether time metrics should be printed immediately after completing
        the learning process.
        :return: None
        """
        self._density_range = self._de_object.extrema
        self._compute_nearest_neighbors_connected()
        self._compute_nearest_neighbors_noise()
        self._compute_connected_components()
        self._label_samples()
        # additional post processing could be done here
        self._performed_clustering = True
        self._time_used = time.time() - start_time
        if print_metrics:
            self.log_util.log_info()
            self.log_util.log_info(
                "=================================================================================================================================")
        self.log_util.log_info("Performed Clustering of '%s' DataSet." % self._scaled_data.get_name())
        if print_metrics:
            self.log_util.log_info("Time used: %.10f seconds" % self._time_used)

    def perform_clustering(self,
                           masslumping: bool = True,
                           lambd: float = 0.0,
                           minimum_level: int = 1,
                           maximum_level: int = 5,
                           print_metrics: bool = True) -> None:
        """Create GridOperation and DensityEstimation objects for the clustering learning process.

        This method is only called once.
        The DensityEstimation objects are mainly used for plotting the combination-scheme later.

        :param masslumping: Optional. Conditional Parameter, which indicates whether masslumping should be enabled for DensityEstimation.
        :param lambd: Optional. Parameter, which adjusts the 'smoothness' of DensityEstimation results.
        :param minimum_level: Optional. Minimum Level of Sparse Grids on which to perform DensityEstimation.
        :param maximum_level: Optional. Maximum Level of Sparse Grids on which to perform DensityEstimation.
        :param print_metrics: Optional. Conditional parameter, which indicates whether time metrics should be printed immediately after completing
        the learning process.
        :return: None
        """
        if self._performed_clustering:
            raise ValueError("Can't perform clustering for the same object twice.")
        start_time = time.time()
        self._clustertinator, self._de_object = self._scaled_data.density_estimation(masslumping=masslumping, lambd=lambd,
                                                                                     minimum_level=minimum_level,
                                                                                     maximum_level=maximum_level, plot_de_dataset=False,
                                                                                     plot_density_estimation=False, plot_combi_scheme=False,
                                                                                     plot_sparsegrid=False)
        self._process_performed_clustering(start_time, print_metrics)

    def perform_clustering_dimension_wise(self,
                                          masslumping: bool = True,
                                          lambd: float = 0.0,
                                          minimum_level: int = 1,
                                          maximum_level: int = 5,
                                          reuse_old_values: bool = False,
                                          numeric_calculation: bool = True,
                                          margin: float = 0.5,
                                          tolerance: float = 0.01,
                                          max_evaluations: int = 256,
                                          modified_basis: bool = False,
                                          boundary: bool = False,
                                          print_metrics: bool = True) -> None:
        """Create dimension-wise GridOperation and DensityEstimation objects for the clustering learning process.

        This method is only called once.
        The DensityEstimation objects are mainly used for plotting the combination-scheme later.

        :param masslumping: Optional. Conditional Parameter which indicates whether masslumping should be enabled for DensityEstimation
        :param lambd: Optional. Parameter which adjusts the 'smoothness' of DensityEstimation results
        :param minimum_level: Optional. Minimum Level of Sparse Grids on which to perform DensityEstimation
        :param maximum_level: Optional. Maximum Level of Sparse Grids on which to perform DensityEstimation
        :param reuse_old_values: Optional.
        :param numeric_calculation: Optional.
        :param margin: Optional.
        :param tolerance: Optional.
        :param max_evaluations: Optional.
        :param modified_basis: Optional.
        :param boundary: Optional.
        :param print_metrics: Optional.
        :return: None
        """
        if self._performed_clustering:
            raise ValueError("Can't perform clustering for the same object twice.")
        start_time = time.time()
        self._clustertinator, self._de_object = self._scaled_data.density_estimation_dimension_wise(masslumping=masslumping,
                                                                                                    lambd=lambd,
                                                                                                    minimum_level=minimum_level,
                                                                                                    maximum_level=maximum_level,
                                                                                                    reuse_old_values=reuse_old_values,
                                                                                                    numeric_calculation=numeric_calculation,
                                                                                                    margin=margin,
                                                                                                    tolerance=tolerance,
                                                                                                    max_evaluations=max_evaluations,
                                                                                                    modified_basis=modified_basis,
                                                                                                    boundary=boundary,
                                                                                                    plot_de_dataset=False,
                                                                                                    plot_density_estimation=False,
                                                                                                    plot_combi_scheme=False,
                                                                                                    plot_sparsegrid=False)
        self._process_performed_clustering(start_time, print_metrics)

    def evaluate(self) -> dict:
        """Evaluate results of all to the original samples assigned labels (if they have original labels to compare to).

        As most of other public methods of Clustering, clustering already has to be performed before this method is called. Otherwise an
        AttributeError is raised.
        In case the original data has no labels to compare to, a ValueError is raised.

        :return: Dictionary of all results.
        """
        if not self._performed_clustering:
            raise AttributeError("Clustering needs to be performed on this object first.")
        omitted, original_data_to_evaluate = self._scaled_data.split_without_labels()
        if original_data_to_evaluate.is_empty():
            raise ValueError("Dataset of this Clustering object doesn't have any labeled samples for comparison.")
        start_time = time.time()
        cannot_evaluate = [(lambda x, y: x[y == 2])(*np.unique(np.where(self._scaled_data[0] == sample)[0], return_counts=True))[0] for sample in
                           omitted[0]]
        computed_data_to_evaluate = self._clustered_data.copy()
        computed_data_to_evaluate.remove_samples(cannot_evaluate)
        number_wrong = 0
        number_original_labels = original_data_to_evaluate.get_number_labels()
        number_computed_labels = computed_data_to_evaluate.get_number_labels()
        # remove wrong clusters
        if number_computed_labels - number_original_labels > 0:
            computed_clusters = computed_data_to_evaluate.split_labels()
            clusters_to_remove = sorted(computed_clusters, key=lambda cluster: cluster.get_length())[:-number_original_labels]
            number_wrong = sum([clus_kept.get_length() for clus_kept in clusters_to_remove])
            data_to_remove = DataSet.list_concatenate(clusters_to_remove)
            definitely_wrong = [(lambda x, y: x[y >= 2])(*np.unique(np.where(computed_data_to_evaluate[0] == sample)[0], return_counts=True))[0]
                                for sample in data_to_remove[0]]
            computed_data_to_evaluate.remove_samples(definitely_wrong)
            original_data_to_evaluate.remove_samples(definitely_wrong)
        for label in computed_data_to_evaluate.get_labels():
            current = np.where(computed_data_to_evaluate[1] == label)[0]
            current_labels = original_data_to_evaluate[1][current]
            unq, count = np.unique(current_labels, return_counts=True)
            equal_label = np.max(count)
            number_wrong += (np.size(current) - equal_label)
        self._time_used += time.time() - start_time
        return {"Time used": self._time_used,
                "Omitted data": omitted,
                "Original data to evaluate": original_data_to_evaluate,
                "Clustered data to evaluate": computed_data_to_evaluate,
                "Wrong mappings": number_wrong,
                "Total mappings": self._scaled_data.get_length(),
                "Percentage correct": (1.0 - (number_wrong / self._scaled_data.get_length())),
                "Percentage correct (str)": "%2.2f%%" % ((1.0 - (number_wrong / self._scaled_data.get_length())) * 100)}

    def print_evaluation(self, print_clusters: bool = True) -> None:
        """Print results of all to the original samples assigned labels (if they have original labels to compare to).

        As most of other public methods of Clustering, clustering already has to be performed before this method is called. Otherwise an
        AttributeError is raised.
        In case the original data has no labels to compare to, a ValueError is raised.

        :return: None
        """
        evaluation = self.evaluate()
        number_wrong = evaluation.get("Wrong mappings")
        omitted = evaluation.get("Omitted data")
        original_data_to_evaluate = evaluation.get("Original data to evaluate")
        clustered_data_to_evaluate = evaluation.get("Clustered data to evaluate")
        self.log_util.log_info(
            "---------------------------------------------------------------------------------------------------------------------------------")
        self.log_util.log_info("Evaluating Clustering object ...")
        if not omitted.is_empty():
            self.log_util.log_info("Omitted some labelless samples of original dataset. Can't compare original labels of these samples with the "
                                   "clustered ones.")
            self.log_util.log_info("Number omitted: %d" % omitted.get_length())
        self.log_util.log_info("Number of wrong mappings: %d" % number_wrong)
        self.log_util.log_info("Number of total mappings: %d" % self._scaled_data.get_length())
        self.log_util.log_info("Percentage of correct mappings: %2.2f%%" % ((1.0 - (number_wrong / self._scaled_data.get_length())) * 100))
        if print_clusters:
            self.log_util.log_info(
                "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
            self.log_util.log_info("Number original clusters: %d" % original_data_to_evaluate.get_number_labels())
            self.log_util.log_info("Number computed clusters: %d" % clustered_data_to_evaluate.get_number_labels())
            self.log_util.log_info("Original data (per label):")
            for label in original_data_to_evaluate.get_labels():
                self.log_util.log_info("%d: %d samples" % (label, np.count_nonzero(original_data_to_evaluate[1] == label)))
            self.log_util.log_info("Clustered data (per cluster):")
            for label in clustered_data_to_evaluate.get_labels():
                self.log_util.log_info("%d: %d samples" % (label, np.count_nonzero(clustered_data_to_evaluate[1] == label)))

    def plot(self,
             plot_original_dataset: bool = False,
             plot_clustered_dataset: bool = False,
             plot_cluster_density_estimation: bool = False,
             plot_cluster_combi_scheme: bool = False,
             plot_cluster_sparsegrid: bool = False,
             plot_nearest_neighbor_graphs: bool = False) -> None:
        """Plot a Clustering object.

        As most of other public methods of Clustering, clustering already has to be performed before this method is called. Otherwise an
        AttributeError is raised.
        The user can specify exactly what to plot with conditional parameters of this method.

        :param plot_original_dataset: Optional. Conditional parameter, which specifies whether the original data set should be plotted. Default False.
        :param plot_clustered_dataset: Optional. Conditional parameter, which specifies whether the original data set with computed clusters should be
        plotted. Default False.
        :param plot_cluster_density_estimation: Optional. Conditional parameter, which specifies whether the density function based on the
        original data set should be plotted. Default False.
        :param plot_cluster_combi_scheme: Optional. Conditional parameter, which specifies whether the combi scheme of the density function should
        be plotted. Default False.
        :param plot_cluster_sparsegrid: Optional. Conditional parameter, which specifies whether the underlying sparse grid of the density function
        should be plotted. Default False.
        :param plot_nearest_neighbor_graphs: Optional. Conditional parameter, which specifies whether the initial and cut nearest neighbor graph
        should be plotted. Default False.
        :return: None
        """
        if not self._performed_clustering:
            raise AttributeError("Clustering needs to be performed on this object first.")
        if plot_original_dataset:
            self._original_data.plot()
        if plot_clustered_dataset:
            self._clustered_data.plot()
        if plot_cluster_density_estimation:
            self._clustertinator.plot(contour=True)
        if plot_cluster_combi_scheme:
            self._clustertinator.print_resulting_combi_scheme(operation=self._de_object)
        if plot_cluster_sparsegrid:
            self._clustertinator.print_resulting_sparsegrid(markersize=20)
        if plot_nearest_neighbor_graphs:
            if not ((self._scaled_data.get_dim() == 2) or (self._scaled_data.get_dim() == 3)):
                warnings.formatwarning = lambda msg, ctg, fname, lineno, file=None, line=None: "%s:%s: %s: %s\n" % (fname, lineno, ctg.__name__, msg)
                warnings.warn("Invalid dimension for plotting. Couldn't plot Nearest Neighbor Graphs.", stacklevel=3)
            else:
                plt.rc('font', size=30)
                plt.rc('axes', titlesize=40)
                plt.rc('figure', figsize=(24.0, 12.0))
                fig = plt.figure()
                ax0 = fig.add_subplot(121) if (self._scaled_data.get_dim() == 2) else fig.add_subplot(121, projection='3d')
                ax1 = fig.add_subplot(122) if (self._scaled_data.get_dim() == 2) else fig.add_subplot(122, projection='3d')
                ax0.set_title("Full_nearest_neighbor_graph")
                ax1.set_title("Cut_nearest_neighbor_graph")
                ax0.title.set_position([0.5, 1.025])
                ax1.title.set_position([0.5, 1.025])
                ax0.grid(True)
                ax1.grid(True)
                ax0.scatter(*zip(*self._scaled_data[0]), s=125)
                ax1.scatter(*zip(*self._scaled_data[0]), s=125)
                data_points = self._scaled_data[0]

                def get_all_axes(edge):
                    ret_val = [[data_points[edge[0]][0], data_points[edge[1]][0]], [data_points[edge[0]][1], data_points[edge[1]][1]]]
                    if self._scaled_data.get_dim() == 3:
                        ret_val.append([data_points[edge[0]][2], data_points[edge[1]][2]])
                    return ret_val

                # plot full graph
                ax0.scatter(*zip(*self._scaled_data[0]), c='r')
                for e in self._all_edges:
                    ax0.plot(*get_all_axes(e), c='r')
                # plot cut graph
                if not self._connected_samples.is_empty():
                    ax1.scatter(*zip(*self._connected_samples[0]), c='r')
                if not self._noise_samples.is_empty():
                    ax1.scatter(*zip(*self._noise_samples[0]), c='y')
                for e in self._remaining_edges:
                    color = 'y' if tuple(e) in self._noise_edges else 'r'
                    ax1.plot(*get_all_axes(e), color)
                ax0.set_xlabel('x')
                ax0.set_ylabel('y')
                ax1.set_xlabel('x')
                ax1.set_ylabel('y')
                if self._scaled_data.get_dim() == 3:
                    ax0.set_zlabel('z')
                    ax1.set_zlabel('z')
                plt.show()
