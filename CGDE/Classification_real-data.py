from sys import path
path.append('../src/')
path.append('../SGDE')
path.append('../SGDE/Datasets')


# sgde tut
from Utils import *

from shutil import copyfile
import os


from sys import path
path.append('../src/')
import DatasetOperation as do
from ErrorCalculator import *
import logging

def prev_level(l, d):
    if l - 2 <= 0:
        return 1
    else:
        return (2**(l-2) - 1) * d + prev_level(l-2, d)

change_log_file('logs/log_classification_real_data')

clear_log()
print_log_info = False
logger.setLevel(logging.INFO)

log_info('--- Classification_eval start ---', True)
for data_set in [2]:

    data_set_name = 'Testset'
    if data_set == 0:
        iris = do.datasets.load_iris()
        sklearn_dataset = (iris.data, iris.target)
        data_set_name = 'iris data set'
    elif data_set == 1:
        wine = do.datasets.load_wine()
        sklearn_dataset = (wine.data, wine.target)
        data_set_name = 'wine data set'
    elif data_set == 2:
        breast_cancer = do.datasets.load_breast_cancer()
        sklearn_dataset = (breast_cancer.data, breast_cancer.target)
        data_set_name = 'breast cancer data set'

    iris = do.datasets.load_iris()
    iris_data = (iris.data, iris.target)
    wine = do.datasets.load_wine()
    wine_data = (wine.data, wine.target)
    breast_cancer = do.datasets.load_breast_cancer()

    test_data_set = do.datasets.make_classification(1000, n_features=2, n_redundant=0, n_clusters_per_class=1,
                                                      n_informative=2, n_classes=3)

    # now we can transform this dataset into a DataSet object and give it an appropriate name
    data = do.DataSet(sklearn_dataset, name=data_set_name)
    data_range = (0.0, 1.0)
    data.scale_range(data_range)

    dim = data.get_dim()
    size = data.get_length()
    tolerance = -1.0

    # use this for quick calculation of grid points used for given level and dimension
    #max_level = 4
    #max_evals = ((2 ** max_level) - 1) * dim - (dim - 1) + (2 ** dim) * prev_level(max_level, dim)

    max_levels = [2, 3]
    start_levels = [x-3 for x in max_levels if x-3 > 1]
    if len(start_levels) == 0:
        start_levels = [2]
    for reuse_old_values in [False]:
        for level_max in max_levels:
            for start_level in start_levels:
                for error_config in [(False, ErrorCalculatorSingleDimVolumeGuided()), (True, ErrorCalculatorSingleDimVolumeGuided()), (True, ErrorCalculatorSingleDimMisclassificationGlobal())]:
                    for rebalancing in [True, False]:
                        for margin in [0.5]:
                            one_vs_others = error_config[0]
                            error_calc = error_config[1]
                            log_info('next iteration', print_log_info)

                            if data_set == 0:
                                log_info('iris data set', print_log_info)
                            elif data_set == 1:
                                log_info('wine data set', print_log_info)
                            elif data_set == 2:
                                log_info('breast cancer data set', print_log_info)

                            log_info('data size: ' + str(size), print_log_info)
                            log_info('data dimension: ' + str(data.get_dim()), print_log_info)
                            t = [i for i, x in enumerate(str(type(error_calc))) if '\'' in x]
                            log_info('error_calculator ' + str(type(error_calc))[t[0]+1:t[-1]], print_log_info)
                            log_info('rebalancing: ' + str(rebalancing), print_log_info)
                            log_info('margin: ' + str(margin), print_log_info)
                            log_info('reuse_old_values: ' + str(reuse_old_values), print_log_info)
                            log_info('one_vs_others ' + str(one_vs_others), print_log_info)


                            # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                            # now let's look at some functions of the DataSet class

                            # # DataSet objects can e.g. be ...
                            data_copy = data.copy()                                              # deepcopied
                            data_copy.scale_range(data_range)                                # scaled
                            # part0, part1 = data_copy.split_pieces(0.5)                           # split
                            # data_copy = part0.concatenate(part1)                                 # concatenated
                            # data_copy.set_name('2nd_Set')                                        # renamed
                            # data_copy.remove_labels(0.2)                                        # freed of some class assignments to samples
                            without_classes, with_classes = data_copy.split_without_labels()    # seperated into samples with and without classes
                            # data_copy.plot()                                                      # plotted

                            data.scale_range(data_range)

                            data_stdCombi = data.copy()
                            data_stdCombi_copy = data_copy.copy()

                            data_dimCombi = data.copy()
                            data_dimCombi_copy = data_copy.copy()


                            # and of course we can perform a regular density estimation on a DataSet object:
                            #de_retval = data_copy.density_estimation(plot_de_dataset=False, plot_sparsegrid=False, plot_density_estimation=True, plot_combi_scheme=True)


                            # initialize Classification object with our original unedited data, 80% of this data is going to be used as learning data which has equally
                            # distributed classes
                            classification = do.Classification(data_stdCombi, split_percentage=0.8, split_evenly=True)

                            # after that we should immediately perform the classification for the learning data tied to the Classification object, since we can't really call any other method before that without raising an error
                            max_level = level_max
                            print('classification max_level', max_level)
                            log_info('classification standardCombi max_level: ' + str(max_level), print_log_info)
                            classification.perform_classification(masslumping=False, lambd=0.0, minimum_level=1, maximum_level=max_level, one_vs_others=one_vs_others, reuse_old_values=reuse_old_values)

                            # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                            # now we can perform some other operations on this classification object

                            # we could e.g. plot its classificators and corresponding density estimations
                            #classification.plot(plot_class_sparsegrid=False, plot_class_combi_scheme=False, plot_class_dataset=True, plot_class_density_estimation=True)

                            # if we already added some testing data to the Classification object (which we did in the initialization process, 20% of samples are testing samples), we can print the current evaluation
                            classification.print_evaluation(print_incorrect_points=False)

                            # we can also add more testing data and print the results immediately
                            #with_classes.set_name("Test_new_data")
                            #classification.test_data(with_classes, print_output=False)

                            # and we can call the Classification object to perform blind classification on a dataset with unknown class assignments to its samples
                            #data_stdCombi_copy.remove_labels(1.0)
                            #calcult_classes = classification(data_stdCombi_copy)

                            # because we used 2D datasets before, we can plot the results to easily see which samples were classified correctly and which not
                            correct_classes = data_stdCombi.copy()
                            correct_classes.scale_range(data_range)
                            #correct_classes.set_name('Correct_Classes')
                            #calcult_classes.set_name('Calculated_Classes')
                            #retfig0 = correct_classes.plot()
                            #retfig1 = calcult_classes.plot()

                            ########################################################################################################################
                            ########################################################################################################################
                            ########################################################################################################################
                            ########################################################################################################################
                            ########################################################################################################################

                            # initialize Classification object with our original unedited data, 80% of this data is going to be used as learning data which has equally
                            # distributed classes
                            classification_dimwise = do.Classification(data_dimCombi, split_percentage=0.8, split_evenly=True)
                            #max_evals = (((2**(max_level-1)) - 1) * dim)

                            max_evals = ((2**max_level) - 1) * dim - (dim - 1) + (2**dim) * prev_level(max_level, dim)
                            print('classification max_evaluations', max_evals)
                            log_info('classification dimwise max_evaluations: ' + str(max_evals), print_log_info)
                            log_info('classification dimwise start level: ' + str(start_level), print_log_info)
                            # after that we should immediately perform the classification for the learning data tied to the Classification object, since we can't really call any other method before that without raising an error
                            if data_set == 0:
                                figure_prefix = 'dimwise_plots/iris'
                            elif data_set == 1:
                                figure_prefix = 'dimwise_plots/wine'
                            elif data_set == 2:
                                figure_prefix = 'dimwise_plots/breast_cancer'
                            classification_dimwise.perform_classification_dimension_wise(masslumping=False,
                                                                                         lambd=0.0,
                                                                                         minimum_level=1, maximum_level=start_level,
                                                                                         reuse_old_values=reuse_old_values,
                                                                                         numeric_calculation=False,
                                                                                         boundary=False,
                                                                                         modified_basis=False,
                                                                                         one_vs_others=one_vs_others,
                                                                                         tolerance=tolerance,
                                                                                         margin=0.5,
                                                                                         rebalancing=rebalancing,
                                                                                         max_evaluations=max_evals,
                                                                                         filename=figure_prefix,
                                                                                         error_calculator=error_calc)

                            # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                            # now we can perform some other operations on this classification object

                            # we could e.g. plot its classificators and corresponding density estimations
                            #classification_dimwise.plot(plot_class_sparsegrid=False, plot_class_combi_scheme=False, plot_class_dataset=False, plot_class_density_estimation=False)

                            # if we already added some testing data to the Classification object (which we did in the initialization process, 20% of samples are testing samples), we can print the current evaluation
                            classification_dimwise.print_evaluation(print_incorrect_points=False)

                            # we can also add more testing data and print the results immediately
                            #with_classes.set_name("Test_new_data")
                            #classification_dimwise.test_data(with_classes, print_output=False)

                            # and we can call the Classification object to perform blind classification on a dataset with unknown class assignments to its samples
                            #data_dimCombi_copy.remove_labels(1.0)
                            #calcult_classes_dimwise = classification(data_dimCombi_copy)

                            # because we used 2D datasets before, we can plot the results to easily see which samples were classified correctly and which not
                            correct_classes_dimwise = data_dimCombi.copy()
                            correct_classes_dimwise.scale_range(data_range)
                            #correct_classes_dimwise.set_name('Correct_Classes_dimwise')
                            #calcult_classes_dimwise.set_name('Calculated_Classes_dimwise')
                            #retfig0 = correct_classes_dimwise.plot()
                            #retfig1 = calcult_classes_dimwise.plot()

                            log_info('iteration end', print_log_info)

log_info('--- Classification_eval end ---', print_log_info)

# make a backup of the log without overwriting old ones
log_backup = 'log_backup'
while os.path.isfile(log_backup):
    log_backup = log_backup + '+'
copyfile(log_filename, log_backup)