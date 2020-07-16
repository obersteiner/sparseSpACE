from sys import path
path.append('../src/')
path.append('../SGDE')
path.append('../SGDE/Datasets')

from Function import *
import numpy as np
import scipy as sp

from spatiallyAdaptiveSingleDimension2 import *
from Function import *
from ErrorCalculator import *

# sgde tut
from GridOperation import *
from StandardCombi import *
from sklearn import datasets
from SGppCompare import plot_comparison
from src.Utils import *

from sklearn.neighbors import KernelDensity

import cProfile
import pstats


from sys import path
path.append('../src/')
import DatasetOperation as do

log_info('--- Classification_eval start ---')
for data_set in range(6):
    log_info('~~~~~~~~~~~~~~~~~~~~~~')
    log_info('~~~ DataSet Change ~~~')
    log_info('~~~~~~~~~~~~~~~~~~~~~~')
    for dimension in [2, 3, 4]:
        log_info('||||||||||||||||||||||||')
        log_info('||| Dimension Change |||')
        log_info('||||||||||||||||||||||||')
        for level_max in [2,3,4,5,6]:
            log_info('########################')
            log_info('### Max Level Change ###')
            log_info('########################')
            for repeats in range(5):
                # generate a Circle-Dataset of size with the sklearn library
                size = 10000
                dim = dimension
                log_info('data size: ' + str(size))
                log_info('data dimension: ' + str(dim))
                if data_set == 0:
                    sklearn_dataset = do.datasets.make_circles(n_samples=size, noise=0.05)
                    log_info('do.datasets.make_circles(n_samples=size, noise=0.05)')
                elif data_set == 1:
                    sklearn_dataset = do.datasets.make_moons(n_samples=size, noise=0.3)
                    log_info('do.datasets.make_moons(n_samples=size, noise=0.3)')
                elif data_set == 2:
                    sklearn_dataset = do.datasets.make_classification(size, n_features=dim, n_redundant=0, n_clusters_per_class=1, n_informative=1, n_classes=2)
                    log_info('do.datasets.make_classification(size, n_features=dim, n_redundant=0, n_clusters_per_class=1, n_informative=1, n_classes=2)')
                elif data_set == 3:
                    sklearn_dataset = do.datasets.make_classification(size, n_features=dim, n_redundant=0, n_clusters_per_class=1, n_informative=2, n_classes=3)
                    log_info('do.datasets.make_classification(size, n_features=dim, n_redundant=0, n_clusters_per_class=1, n_informative=2, n_classes=3)')
                elif data_set == 4:
                    sklearn_dataset = do.datasets.make_blobs(n_samples=size, n_features=dim, centers=6)
                    log_info('do.datasets.make_blobs(n_samples=size, n_features=dim centers=6)')
                elif data_set == 5:
                    sklearn_dataset = do.datasets.make_gaussian_quantiles(n_samples=size, n_features=dim, n_classes=6)
                    log_info('do.datasets.make_gaussian_quantiles(n_samples=size, n_features=dim, n_classes=6)')


                # now we can transform this dataset into a DataSet object and give it an appropriate name
                data = do.DataSet(sklearn_dataset, name='Testset')

                # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                # now let's look at some functions of the DataSet class

                # # DataSet objects can e.g. be ...
                data_copy = data.copy()                                              # deepcopied
                data_copy.scale_range((0.005, 0.995))                                # scaled
                # part0, part1 = data_copy.split_pieces(0.5)                           # split
                # data_copy = part0.concatenate(part1)                                 # concatenated
                # data_copy.set_name('2nd_Set')                                        # renamed
                # data_copy.remove_classes(0.2)                                        # freed of some class assignments to samples
                without_classes, with_classes = data_copy.split_without_classes()    # seperated into samples with and without classes
                # data_copy.plot()                                                     # plotted

                # and of course we can perform a regular density estimation on a DataSet object:
                #de_retval = data_copy.density_estimation(plot_de_dataset=False, plot_sparsegrid=False, plot_density_estimation=True, plot_combi_scheme=True)


                # initialize Classification object with our original unedited data, 80% of this data is going to be used as learning data which has equally
                # distributed classes
                classification = do.Classification(data, split_percentage=0.8, split_evenly=True)

                # after that we should immediately perform the classification for the learning data tied to the Classification object, since we can't really call any other method before that without raising an error
                max_level = 5
                print('classification max_level', max_level)
                log_info('classification standardCombi max_level: ' + str(max_level))
                classification.perform_classification(masslumping=False, lambd=0.0, minimum_level=1, maximum_level=max_level)

                # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                # now we can perform some other operations on this classification object

                # we could e.g. plot its classificators and corresponding density estimations
                #classification.plot(plot_class_sparsegrid=False, plot_class_combi_scheme=False, plot_class_dataset=True, plot_class_density_estimation=True)

                # if we already added some testing data to the Classification object (which we did in the initialization process, 20% of samples are testing samples), we can print the current evaluation
                classification.print_evaluation()

                # we can also add more testing data and print the results immediately
                with_classes.set_name("Test_new_data")
                classification.test_data(with_classes, print_output=False)

                # and we can call the Classification object to perform blind classification on a dataset with unknown class assignments to its samples
                data_copy.remove_classes(1.0)
                calcult_classes = classification(data_copy)

                # because we used 2D datasets before, we can plot the results to easily see which samples were classified correctly and which not
                correct_classes = data.copy()
                correct_classes.scale_range((0.005, 0.995))
                correct_classes.set_name('Correct_Classes')
                calcult_classes.set_name('Calculated_Classes')
                #retfig0 = correct_classes.plot()
                #retfig1 = calcult_classes.plot()

                ########################################################################################################################
                ########################################################################################################################
                ########################################################################################################################
                ########################################################################################################################
                ########################################################################################################################

                # initialize Classification object with our original unedited data, 80% of this data is going to be used as learning data which has equally
                # distributed classes
                classification_dimwise = do.Classification(data, split_percentage=0.8, split_evenly=True)
                max_evals = (((2**max_level) - 1) * dim)
                print('classification max_evaluations', max_evals)
                log_info('classification dimwise max_evaluations: ' + str(max_evals))
                # after that we should immediately perform the classification for the learning data tied to the Classification object, since we can't really call any other method before that without raising an error
                classification_dimwise.perform_classification_dimension_wise(_masslumping=False, _lambd=0.0, _minimum_level=1, _maximum_level=2,
                                                                     _reuse_old_values=True, _numeric_calculation=False,
                                                                     _boundary=False, _modified_basis=False,
                                                                     _tolerance=0.0, _margin=0.5, _max_evaluations=max_evals)

                # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                # now we can perform some other operations on this classification object

                # we could e.g. plot its classificators and corresponding density estimations
                #classification_dimwise.plot(plot_class_sparsegrid=False, plot_class_combi_scheme=False, plot_class_dataset=False, plot_class_density_estimation=False)

                # if we already added some testing data to the Classification object (which we did in the initialization process, 20% of samples are testing samples), we can print the current evaluation
                classification_dimwise.print_evaluation()

                # we can also add more testing data and print the results immediately
                with_classes.set_name("Test_new_data")
                classification_dimwise.test_data(with_classes, print_output=False)

                # and we can call the Classification object to perform blind classification on a dataset with unknown class assignments to its samples
                data_copy.remove_classes(1.0)
                calcult_classes_dimwise = classification(data_copy)

                # because we used 2D datasets before, we can plot the results to easily see which samples were classified correctly and which not
                correct_classes_dimwise = data.copy()
                correct_classes_dimwise.scale_range((0.005, 0.995))
                correct_classes_dimwise.set_name('Correct_Classes_dimwise')
                calcult_classes_dimwise.set_name('Calculated_Classes_dimwise')
                #retfig0 = correct_classes_dimwise.plot()
                #retfig1 = calcult_classes_dimwise.plot()

log_info('--- Classification_eval end ---')