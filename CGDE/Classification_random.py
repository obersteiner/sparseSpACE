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

change_log_file('logs/log_classification_random')

clear_log()
print_log_info = False
logger.setLevel(logging.INFO)

dim = 4
max_level = 8
test = ((2 ** max_level) - 1) * dim - (dim - 1) + (2 ** dim) * prev_level(max_level, dim)
dim = 5
max_level = 7
test2 = ((2 ** max_level) - 1) * dim - (dim - 1) + (2 ** dim) * prev_level(max_level, dim)

tolerance = -1.0

log_info('--- Classification_eval start ---', True)
for dimension in [2, 3, 4, 5]:

    # generate a Circle-Dataset of size with the sklearn library
    size = 25000
    dim = dimension
    sklearn_dataset = do.datasets.make_classification(size, n_features=dim, n_redundant=0, n_clusters_per_class=1,
                                                      n_informative=2, n_classes=3)

    data = do.DataSet(sklearn_dataset, name='gaussian quantiles')
    data_range = (0.0, 1.0)
    data.scale_range(data_range)

    reuse_old_values = False
    dimWiseInitialized = False

    data_copy = data.copy()  # deepcopied
    data_copy.scale_range(data_range)  # scaled
    without_classes, with_classes = data_copy.split_without_labels()  # seperated into samples with and without classes                                                    # plotted

    data.scale_range(data_range)

    data_stdCombi = data.copy()
    data_stdCombi_copy = data_copy.copy()

    data_dimCombi = data.copy()
    data_dimCombi_copy = data_copy.copy()

    max_levels = [2, 3, 4, 5, 6]
    start_levels = [x - 3 for x in max_levels if 1 < x - 3 < 4]
    if len(start_levels) == 0:
        start_levels = [2]
    #for level_max in max_levels:
    for margin in [0.5]:
        for start_level in start_levels:
            for error_config in [(False, ErrorCalculatorSingleDimVolumeGuided()), (True, ErrorCalculatorSingleDimVolumeGuided()), (True, ErrorCalculatorSingleDimMisclassificationGlobal())]:
                for rebalancing in [True, False]:
                    dimWiseInitialized = False
                    #for margin in [0.5]:
                    for level_max in max_levels:
                        one_vs_others = error_config[0]
                        error_calc = error_config[1]
                        log_info('next iteration', print_log_info)

                        log_info(
                            'do.datasets.make_classification(size, n_features=dim, n_redundant=0, n_clusters_per_class=1, n_informative=2, n_classes=3)',
                            print_log_info)
                        log_info('data size: ' + str(size), print_log_info)
                        log_info('data dimension: ' + str(data.get_dim()), print_log_info)
                        t = [i for i, x in enumerate(str(type(error_calc))) if '\'' in x]
                        log_info('error_calculator ' + str(type(error_calc))[t[0]+1:t[-1]], print_log_info)
                        log_info('rebalancing: ' + str(rebalancing), print_log_info)
                        log_info('margin: ' + str(margin), print_log_info)
                        log_info('one_vs_others ' + str(one_vs_others), print_log_info)

                        classification = do.Classification(data_stdCombi, split_percentage=0.8, split_evenly=True)

                        max_level = level_max
                        print('classification max_level', max_level)
                        log_info('classification standardCombi max_level: ' + str(max_level), print_log_info)
                        classification.perform_classification(masslumping=False, lambd=0.0, minimum_level=1, maximum_level=max_level, one_vs_others=one_vs_others, reuse_old_values=reuse_old_values)

                        classification.print_evaluation(print_incorrect_points=False)

                        ########################################################################################################################
                        ########################################################################################################################
                        ########################################################################################################################
                        ########################################################################################################################
                        ########################################################################################################################

                        if not dimWiseInitialized:
                            classification_dimwise = do.Classification(data_dimCombi, split_percentage=0.8, split_evenly=True)
                        #max_evals = (((2**(max_level-1)) - 1) * dim)

                        max_evals = ((2**max_level) - 1) * dim - (dim - 1) + (2**dim) * prev_level(max_level, dim)
                        print('classification max_evaluations', max_evals)
                        log_info('classification dimwise max_evaluations: ' + str(max_evals), print_log_info)
                        log_info('classification dimwise start level: ' + str(start_level), print_log_info)
                        log_info('classification dimwise rebalance ' + str(rebalancing), print_log_info)
                        log_info('classification dimwise margin ' + str(margin), print_log_info)

                        figure_prefix = 'dimwise_plots/gaussian_quantiles'

                        if not dimWiseInitialized:
                            classification_dimwise.perform_classification_dimension_wise(masslumping=False,
                                                                                         lambd=0.0,
                                                                                         minimum_level=1,
                                                                                         maximum_level=start_level,
                                                                                         reuse_old_values=reuse_old_values,
                                                                                         numeric_calculation=False,
                                                                                         boundary=False,
                                                                                         modified_basis=False,
                                                                                         one_vs_others=one_vs_others,
                                                                                         tolerance=tolerance,
                                                                                         margin=margin,
                                                                                         rebalancing=rebalancing,
                                                                                         max_evaluations=max_evals,
                                                                                         filename=figure_prefix,
                                                                                         error_calculator=error_calc)
                            dimWiseInitialized = True
                        else:
                            classification_dimwise.continue_dimension_wise_refinement(tolerance=tolerance, max_evaluations=max_evals)


                        classification_dimwise.print_evaluation(print_incorrect_points=False)

                        log_info('iteration end', print_log_info)

log_info('--- Classification_eval end ---', print_log_info)

# make a backup of the log without overwriting old ones
log_backup = 'log_sg_backup'
while os.path.isfile(log_backup):
    log_backup = log_backup + '+'
copyfile(log_filename, log_backup)