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
import DEMachineLearning as do
from ErrorCalculator import *
import logging

def prev_level(l, d):
    if l - 2 <= 0:
        return 1
    else:
        return (2**(l-2) - 1) * d + prev_level(l-2, d)

change_log_file('logs/log_classification_various')

clear_log()
logUtil.set_log_level(log_levels.INFO)

logUtil.log_info('--- Classification_eval start ---')
for data_set in [2]:
    for dimension in [3, 4, 5]:

        # generate a Circle-Dataset of size with the sklearn library
        size = 10000
        dim = dimension
        if data_set == 0:
            sklearn_dataset = do.datasets.make_classification(size, n_features=dim, n_redundant=0, n_clusters_per_class=1, n_informative=2, n_classes=3)
        elif data_set == 1:
            sklearn_dataset = do.datasets.make_blobs(n_samples=size, n_features=dim, centers=3)
        elif data_set == 2:
            sklearn_dataset = do.datasets.make_gaussian_quantiles(n_samples=size, n_features=dim, n_classes=6)


        # now we can transform this dataset into a DataSet object and give it an appropriate name
        data = do.DataSet(sklearn_dataset, name='Testset')
        data_range = (0.0, 1.0)
        data.scale_range(data_range)

        reuse_old_values = False

        max_levels = [2,3,4,5,6]
        start_levels = [x - 3 for x in max_levels if 1 < x - 3 < 4]
        if len(start_levels) == 0:
            start_levels = [2]
        for level_max in max_levels:
            for start_level in start_levels:
                for error_config in [(False, ErrorCalculatorSingleDimVolumeGuided()), (True, ErrorCalculatorSingleDimVolumeGuided()), (True, ErrorCalculatorSingleDimMisclassificationGlobal())]:
                    for rebalancing in [True, False]:
                        one_vs_others = error_config[0]
                        error_calc = error_config[1]
                        logUtil.log_info('next iteration')

                        if data_set == 0:
                            logUtil.log_info('do.datasets.make_circles(n_samples=size, noise=0.05)')
                        elif data_set == 1:
                            logUtil.log_info('do.datasets.make_moons(n_samples=size, noise=0.3)')
                        elif data_set == 2:
                            logUtil.log_info('do.datasets.make_classification(size, n_features=dim, n_redundant=0, n_clusters_per_class=1, n_informative=1, n_classes=2)')
                        elif data_set == 3:
                            logUtil.log_info('do.datasets.make_classification(size, n_features=dim, n_redundant=0, n_clusters_per_class=1, n_informative=2, n_classes=3)')
                        elif data_set == 4:
                            logUtil.log_info('do.datasets.make_blobs(n_samples=size, n_features=dim centers=6)')
                        elif data_set == 5:
                            logUtil.log_info('do.datasets.make_gaussian_quantiles(n_samples=size, n_features=dim, n_classes=6)')

                        logUtil.log_info('data size: ' + str(size))
                        logUtil.log_info('data dimension: ' + str(data.get_dim()))
                        t = [i for i, x in enumerate(str(type(error_calc))) if '\'' in x]
                        logUtil.log_info('error_calculator ' + str(type(error_calc))[t[0]+1:t[-1]])
                        logUtil.log_info('one_vs_others ' + str(one_vs_others))

                        data_copy = data.copy()                                              # deepcopied
                        data_copy.scale_range(data_range)                                # scaled
                        without_classes, with_classes = data_copy.split_without_labels()    # seperated into samples with and without classes                                                 # plotted

                        data.scale_range(data_range)

                        data_stdCombi = data.copy()
                        data_stdCombi_copy = data_copy.copy()

                        data_dimCombi = data.copy()
                        data_dimCombi_copy = data_copy.copy()

                        classification = do.Classification(data_stdCombi, split_percentage=0.8, split_evenly=True)

                        max_level = level_max
                        print('classification max_level', max_level)
                        logUtil.log_info('classification standardCombi max_level: ' + str(max_level))
                        classification.perform_classification(masslumping=False, lambd=0.0, minimum_level=1, maximum_level=max_level, one_vs_others=one_vs_others, reuse_old_values=reuse_old_values)

                        classification.print_evaluation(print_incorrect_points=False)

                        #correct_classes = data_stdCombi.copy()
                        #correct_classes.scale_range(data_range)
                        #correct_classes.set_name('Correct_Classes')
                        #calcult_classes.set_name('Calculated_Classes')
                        #retfig0 = correct_classes.plot()
                        #retfig1 = calcult_classes.plot()

                        ########################################################################################################################
                        ########################################################################################################################
                        ########################################################################################################################
                        ########################################################################################################################
                        ########################################################################################################################

                        classification_dimwise = do.Classification(data_dimCombi, split_percentage=0.8, split_evenly=True)
                        #max_evals = (((2**(max_level-1)) - 1) * dim)

                        max_evals = ((2**max_level) - 1) * dim - (dim - 1) + (2**dim) * prev_level(max_level, dim)
                        print('classification max_evaluations', max_evals)
                        logUtil.log_info('classification dimwise max_evaluations: ' + str(max_evals))
                        logUtil.log_info('classification dimwise start level: ' + str(start_level))

                        if data_set == 0:
                            figure_prefix = 'dimwise_plots/circles'
                        elif data_set == 1:
                            figure_prefix = 'dimwise_plots/moons'
                        elif data_set == 2:
                            figure_prefix = 'dimwise_plots/classification'
                        elif data_set == 3:
                            figure_prefix = 'dimwise_plots/classification'
                        elif data_set == 4:
                            figure_prefix = 'dimwise_plots/blobs'
                        elif data_set == 5:
                            figure_prefix = 'dimwise_plots/gaussian_quantiles'
                        classification_dimwise.perform_classification_dimension_wise(masslumping=False,
                                                                                     lambd=0.0,
                                                                                     minimum_level=1, maximum_level=start_level,
                                                                                     reuse_old_values=reuse_old_values,
                                                                                     numeric_calculation=False,
                                                                                     boundary=False,
                                                                                     modified_basis=False,
                                                                                     one_vs_others=one_vs_others,
                                                                                     tolerance=0.05,
                                                                                     margin=0.5,
                                                                                     rebalancing=rebalancing,
                                                                                     max_evaluations=max_evals,
                                                                                     error_calculator=error_calc)

                        classification_dimwise.print_evaluation(print_incorrect_points=False)

                        #correct_classes_dimwise = data_dimCombi.copy()
                        #correct_classes_dimwise.scale_range(data_range)
                        #correct_classes_dimwise.set_name('Correct_Classes_dimwise')
                        #calcult_classes_dimwise.set_name('Calculated_Classes_dimwise')
                        #retfig0 = correct_classes_dimwise.plot()
                        #retfig1 = calcult_classes_dimwise.plot()

                        logUtil.log_info('iteration end')

logUtil.log_info('--- Classification_eval end ---')

# make a backup of the log without overwriting old ones
log_backup = 'log_sg_backup'
while os.path.isfile(log_backup):
    log_backup = log_backup + '+'
copyfile(log_filename, log_backup)