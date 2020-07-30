#%matplotlib inline
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

def prev_level(l, d):
    if l - 2 <= 0:
        return 1
    else:
        return (2**(l-2) - 1) * d + prev_level(l-2, d)
# dim = 2
# test1 = 1
# test2 = ((2**2) - 1) * dim - (dim - 1) + (2**dim) * prev_level(3, dim)
# test3 = ((2**3) - 1) * dim - (dim - 1) + (2**dim) * prev_level(4, dim)
# test4 = ((2**4) - 1) * dim - (dim - 1) + (2**dim) * prev_level(5, dim)
# test5 = ((2**5) - 1) * dim - (dim - 1) + (2**dim) * prev_level(5, dim)
# max_eval = ((2**max_level) - 1) * dim - (dim - 1) + (2**dim) * prev_level(max_level, dim)
# print('stop')



def scale_data(data, dim, scale):
    scaler = preprocessing.MinMaxScaler(feature_range=(scale[0], scale[1]))
    if (isinstance(data, tuple)):
        scaler.fit(data[0])
        data = scaler.transform(data[0])
    else:
        if (dim > 1):
            scaler.fit(data)
        else:
            data = data.reshape(-1, 1)
            scaler.fit(data)
        data = scaler.transform(data)
    return data


def plot_dataset(d, dim, filename: str = None):
    fontsize = 30
    data = d[:, :dim]
    plt.rcParams.update({'font.size': fontsize})
    fig = plt.figure(figsize=(10, 10))
    if dim == 2:
        ax = fig.add_subplot(1, 1, 1)
        x, y = zip(*data)
        ax.scatter(x, y, s=125)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title("M = %d" % len(data))

    elif dim == 3:
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        x, y, z = zip(*data)
        ax.scatter(x, y, z, s=125)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title("#points = %d" % len(data))

    else:
        print("Cannot print data of dimension > 2")

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()
    # reset fontsize to default so it does not affect other figures
    plt.rcParams.update({'font.size': plt.rcParamsDefault.get('font.size')})

clear_log()
logger.setLevel(logging.INFO)
log_info('--- DimWiseClassification begin ---')

# dimension of the problem
dim = 2
print('data set dimension: ', dim)
# define number of samples
size = 500
print('data set size: ', size)

# define boundaries
a = np.zeros(dim)
b = np.ones(dim)

# choose data set type
data_sets = ['floats', 'std exponential', 'std normal', 'multi normal', 'line', 'cross', 'moon', 'circle',
             'multi normal class', 'moon class', 'checkerboard class']
data_set = data_sets[7]
print('chosen data set:', data_set)
scale = [0.0000000001, 1.0]
print('chosen scaling: ', scale)

###################### choose grid parameters
# define lambda
lambd = 0.02
print('DensityEstimation lambda:', lambd)
# use modified basis function
modified_basis = False
print('modified_basis:', modified_basis)
# put points on the boundary
boundary = False
print('points on boundary:', boundary)
# reuse older R matrix values
reuse_old_values = True
print('reuse old values: ', reuse_old_values)
# choose between numeric and analytic calculation
numeric_calculation = False
print('numeric_calculation: ', numeric_calculation)
# define level of standard combigrid
minimum_level, maximum_level = 1, 4
print('max level of standard combirid:', minimum_level, ' : ', maximum_level)
# define starting level of dimension wise combigrid
lmin, lmax = 1, 2
print('lim/lmax of dimWise grid: ', lmin, ' : ', lmax)
# error tolerance
tolerance = 0.00
print('error tolerance:', tolerance)
# error margin
margin = 0.5
print('error margin: ', margin)
# maximum amount of new grid_points
max_evaluations = 3000
print('max evaluations for dimWise:', max_evaluations)
# plot the resulting combi-scheme with each refinement
do_plot = False
print('refinement plotting:', do_plot)

# kde parameters
kde_bandwidth = 0.05
print('kde_bandwidth:', kde_bandwidth)

data = None
class_signs = None
############ DATASETS
if data_set == 'floats':
    # define data (https://docs.scipy.org/doc/numpy-1.14.0/reference/routines.random.html)
    # random floats
    data = np.random.random((size, dim))
elif data_set == 'std exponential':
    # samples from the standard exponential distribution.
    data = np.random.standard_exponential((size, dim))
elif data_set == 'std normal':
    # samples from the standard normal distribution
    data = np.random.standard_normal((size, dim))
elif data_set == 'multi normal':
    # multivariate normal distribution
    mean = np.array([0.0] * dim)
    sigma = np.array([0.25]*dim)
    cov = np.diag(sigma**2)
    data = np.random.multivariate_normal(mean, cov, size)
elif data_set == 'uniform':
    # uniform distribution
    data = np.random.uniform(0.0, 1.0, [size, 2])  # whole domain
elif data_set == 'line':
    # Line
    uni = np.random.uniform(0.0, 1.0, size)
    #constant = np.ones(size) * 0.5
    constant = np.random.uniform(0.45, 0.55, size)
    data = np.vstack((constant, uni)).T
elif data_set == 'cross':
    # Cross
    uni = np.random.uniform(0.0, 1.0, int(size / 2))
    #constant = np.ones(int(size / 2)) * 0.5
    cross_const_dom = [0.49, 0.51]
    constant = np.random.uniform(cross_const_dom[0], cross_const_dom[1], int(size / 2))
    data = np.vstack((np.hstack((constant, uni)), np.hstack((uni, constant)))).T
elif data_set == 'moon':
    # scikit learn datasets
    data = datasets.make_moons(size, noise=0.1)[0]
elif data_set == 'circle':
    data = datasets.make_circles(size, noise=0.1)[0]
elif data_set == 'multi normal class':
    # multivariate normal distribution
    mean_a = np.array([-0.5] * dim)
    mean_b = np.array([+0.5] * dim)
    sigma = np.array([0.25]*dim)
    cov = np.diag(sigma**2)
    class_a = np.random.multivariate_normal(mean_a, cov, int(size/2))
    class_b = np.random.multivariate_normal(mean_b, cov, int(size/2))
    data = np.vstack((class_a, class_b))

    a_sign = np.ones(int(size/2))
    b_sign = np.ones(int(size/2)) * -1.0
    class_signs = np.hstack((a_sign, b_sign))
elif data_set == 'moon class':
    ret = datasets.make_moons(size, noise=0.1)
    data = ret[0]
    class_signs = np.array([-1 if p == 0 else 1 for p in ret[1]])

elif data_set == 'checkerboard class':
    checkerboard_shape = (size, dim)
    checkerboard_class_number = 2
    ret = datasets.make_checkerboard(checkerboard_shape, checkerboard_class_number, noise=0.1, minval=0.0, maxval=1.0)
    data = ret[0]
    class_signs = ret[1][0]
    class_signs = np.array([-1 if not p else 1 for p in class_signs])


if data_set is not 'line':
    data = scale_data(data, dim, scale)

print('plot of data set: ')
#plot_dataset(data, dim, 'dataPlot_'+data_set)

# csv dataset file
#data = "Datasets/faithful.csv"
# SGpp values for dataset
# values = "Values/Circles_level_4_lambda_0.0.csv"

########### GRID EVALUATIONS
timings = {}  # pass this dict to the operation and grid scheme to collect execution time information

newGrid = GlobalTrapezoidalGrid(a=np.zeros(dim), b=np.ones(dim), modified_basis=modified_basis, boundary=boundary)

if 'class' in data_set:
    # errorOperator = ErrorCalculatorSingleDimMisclassification()
    errorOperator = ErrorCalculatorSingleDimMisclassificationGlobal()
else:
    errorOperator = ErrorCalculatorSingleDimVolumeGuided()


# define operation to be performed
op = DensityEstimation(data, dim, grid=newGrid, lambd=lambd, classes=class_signs, reuse_old_values=reuse_old_values, numeric_calculation=numeric_calculation)
# create the combiObject and initialize it with the operation
SASD = SpatiallyAdaptiveSingleDimensions2(a, b, operation=op, margin=margin, timings=timings, rebalancing=False)
if do_plot:
    print("Plot of dataset:")
    op.plot_dataset(filename='dimWise_'+data_set+'_dataSet')
# perform the density estimation operation, has to be done before the printing and plotting
# cProfile.run('SASD.performSpatiallyAdaptiv(lmin, lmax, errorOperator, tolerance, max_evaluations=max_evaluations, do_plot=do_plot)',
#              filename='DimWiseAdaptivProfile.txt')
# p_stat = pstats.Stats('DimWiseAdaptivProfile.txt')
#
# p_stat.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(100)


from sys import path
path.append('../src/')
import DatasetOperation as do


# generate a Circle-Dataset of size with the sklearn library
size = 10000
dim = 5
one_vs_others = False
log_info('data size: ' + str(size))
log_info('data dimension: ' + str(dim))
log_info('one vs others: ' + str(one_vs_others))
# sklearn_dataset = do.datasets.make_circles(n_samples=size, noise=0.05)
# sklearn_dataset = do.datasets.make_moons(n_samples=size, noise=0.3)
# sklearn_dataset = do.datasets.make_classification(size, n_features=dim, n_redundant=0, n_clusters_per_class=1, n_informative=1, n_classes=2)
# sklearn_dataset = do.datasets.make_classification(size, n_features=dim, n_redundant=0, n_clusters_per_class=1, n_informative=2, n_classes=3)
# sklearn_dataset = do.datasets.make_blobs(n_samples=size, n_features=dim, centers=6)
sklearn_dataset = do.datasets.make_gaussian_quantiles(n_samples=size, n_features=dim, n_classes=6)

log_info('used data set: ' + 'do.datasets.make_classification(size, n_features=dim, n_redundant=0, n_clusters_per_class=1, n_informative=2, n_classes=3)')

# now we can transform this dataset into a DataSet object and give it an appropriate name
data = do.DataSet(sklearn_dataset, name='Testset')
data.plot()
data_range = (0.0, 1.0)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# now let's look at some functions of the DataSet class

# DataSet objects can e.g. be ...
data_copy = data.copy()                                              # deepcopied
data_copy.scale_range(data_range)                                # scaled
part0, part1 = data_copy.split_pieces(0.5)                           # split
data_copy = part0.concatenate(part1)                                 # concatenated
data_copy.set_name('2nd_Set')                                        # renamed
data_copy.remove_labels(0.2)                                        # freed of some class assignments to samples
without_classes, with_classes = data_copy.split_without_labels()    # seperated into samples with and without classes
#data_copy.plot()                                                     # plotted

#data.plot()
lab = data.get_labels()
dat = data.get_data()

# for i in range(len(lab)):
#     print_copy = data.copy()
#     for j in range(len(lab)):
#         if j != i:
#             to_remove = np.argwhere(print_copy[1] == lab[j]).flatten()
#             print_copy.remove_samples(to_remove)
#     print_copy.plot()

# data.scale_range(data_range)
data.plot()
## plot single classes
# for i in range(len(lab)):
#     print_copy = data.copy()
#     for j in range(len(lab)):
#         if j != i:
#             to_remove = np.argwhere(print_copy[1] == lab[j]).flatten()
#             print_copy.remove_samples(to_remove)
#     print_copy.plot()

# and of course we can perform a regular density estimation on a DataSet object:
#de_retval = data_copy.density_estimation(plot_de_dataset=False, plot_sparsegrid=False, plot_density_estimation=True, plot_combi_scheme=True)

# initialize Classification object with our original unedited data, 80% of this data is going to be used as learning data which has equally
# distributed classes
classification = do.Classification(data, split_percentage=0.8, split_evenly=True)

# after that we should immediately perform the classification for the learning data tied to the Classification object, since we can't really call any other method before that without raising an error
max_level = 6
print('classification max_level', max_level)
log_info('classification standardCombi max_level: ' + str(max_level))
classification.perform_classification(masslumping=False,
                                      lambd=0.0,
                                      minimum_level=1,
                                      maximum_level=max_level,
                                      one_vs_others=one_vs_others,
                                      pre_scaled_data=True)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# now we can perform some other operations on this classification object


# we could e.g. plot its classificators and corresponding density estimations
classification.plot(plot_class_sparsegrid=False, plot_class_combi_scheme=False, plot_class_dataset=False, plot_class_density_estimation=True)

# if we already added some testing data to the Classification object (which we did in the initialization process, 20% of samples are testing samples), we can print the current evaluation
classification.print_evaluation()

# we can also add more testing data and print the results immediately
#with_classes.set_name("Test_new_data")
#classification.test_data(with_classes, print_output=False)

# and we can call the Classification object to perform blind classification on a dataset with unknown class assignments to its samples
#data_copy.remove_labels(1.0)
classif_data = classification.get_data()
classif_data.remove_labels(1.0)
calcult_classes = classification(classif_data)

# because we used 2D datasets before, we can plot the results to easily see which samples were classified correctly and which not
correct_classes = data.copy()
correct_classes.scale_range(data_range)
correct_classes.set_name('Correct_Classes')
calcult_classes.set_name('Calculated_Classes')
retfig0 = correct_classes.plot()
retfig1 = calcult_classes.plot()

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

# initialize Classification object with our original unedited data, 80% of this data is going to be used as learning data which has equally
# distributed classes
classification_dimwise = do.Classification(data, split_percentage=0.8, split_evenly=True)

#max_evals = (((2**max_level) - 1) * dim)
max_evals = ((2**max_level) - 1) * dim - (dim - 1) + (2**dim) * prev_level(max_level, dim)
print('classification max_evaluations', max_evals)
log_info('classification dimwise max_evaluations: ' + str(max_evals))
# after that we should immediately perform the classification for the learning data tied to the Classification object, since we can't really call any other method before that without raising an error
classification_dimwise.perform_classification_dimension_wise(masslumping=False, lambd=0.0, minimum_level=1, maximum_level=3,
                                                     reuse_old_values=True, numeric_calculation=False,
                                                     boundary=boundary, modified_basis=modified_basis, one_vs_others=False,
                                                     tolerance=tolerance, margin=margin, max_evaluations=max_evals,
                                                             pre_scaled_data=True, filename='DimWiseClassification_')

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# now we can perform some other operations on this classification object


# we could e.g. plot its classificators and corresponding density estimations
classification_dimwise.plot(plot_class_sparsegrid=False, plot_class_combi_scheme=False, plot_class_dataset=False, plot_class_density_estimation=True)

# if we already added some testing data to the Classification object (which we did in the initialization process, 20% of samples are testing samples), we can print the current evaluation
classification_dimwise.print_evaluation()

# we can also add more testing data and print the results immediately
#with_classes.set_name("Test_new_data")
#classification_dimwise.test_data(with_classes, print_output=False)

# and we can call the Classification object to perform blind classification on a dataset with unknown class assignments to its samples
data_copy.remove_labels(1.0)
classif_dimwise_data = classification.get_data()
classif_dimwise_data.remove_labels(1.0)
calcult_classes_dimwise = classification_dimwise(classif_dimwise_data)

# because we used 2D datasets before, we can plot the results to easily see which samples were classified correctly and which not
correct_classes_dimwise = data.copy()
correct_classes_dimwise.scale_range(data_range)
correct_classes_dimwise.set_name('Correct_Classes_dimwise')
calcult_classes_dimwise.set_name('Calculated_Classes_dimwise')
retfig0 = correct_classes_dimwise.plot()
retfig1 = calcult_classes_dimwise.plot()


log_info('--- DimWiseClassification end ---')