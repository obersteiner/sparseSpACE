#%matplotlib inline
from sys import path
path.append('../src/')
path.append('../SGDE')
path.append('../SGDE/Datasets')

from ErrorCalculator import *

from GridOperation import *
from StandardCombi import *
from src.Utils import *

from sys import path
path.append('../src/')
import DatasetOperation as do



def prev_level(l, d):
    if l - 2 <= 0:
        return 1
    else:
        return (2**(l-2) - 1) * d + prev_level(l-2, d)


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


# generate a Circle-Dataset of size with the sklearn library
size = 10000
dim = 2
#error_config = (ErrorCalculatorSingleDimVolumeGuided(), False)
error_config = (ErrorCalculatorSingleDimMisclassificationGlobal(), True)
error_calculator = error_config[0]
one_vs_others = error_config[1]

log_info('data size: ' + str(size))
log_info('data dimension: ' + str(dim))
log_info('one vs others: ' + str(one_vs_others))
log_info('error_calculator: ' + str(type(error_calculator)))
#sklearn_dataset = do.datasets.make_circles(n_samples=size, noise=0.05)
#data_set_name = 'Circles'
#sklearn_dataset = do.datasets.make_moons(n_samples=size, noise=0.3)
#data_set_name = 'Two Moons'
# sklearn_dataset = do.datasets.make_moons(n_samples=size, noise=0.3)
# sklearn_dataset = do.datasets.make_classification(size, n_features=dim, n_redundant=0, n_clusters_per_class=1, n_informative=1, n_classes=2)
#sklearn_dataset = do.datasets.make_classification(size, n_features=dim, n_redundant=0, n_clusters_per_class=1, n_informative=2, n_classes=3)
#data_set_name = 'Random Classes'
#sklearn_dataset = do.datasets.make_blobs(n_samples=size, n_features=dim, centers=6)
#data_set_name = 'Blobs'
sklearn_dataset = do.datasets.make_gaussian_quantiles(n_samples=size, n_features=dim, n_classes=4)
data_set_name = 'Gaussian Quantiles'

#breast_cancer = do.datasets.load_breast_cancer()
#sklearn_dataset = (breast_cancer.data, breast_cancer.target)

log_info('used data set: ' + 'do.datasets.make_classification(size, n_features=dim, n_redundant=0, n_clusters_per_class=1, n_informative=2, n_classes=3)')

# now we can transform this dataset into a DataSet object and give it an appropriate name
data = do.DataSet(sklearn_dataset, name=data_set_name)
#data.plot()
data_range = (0.0, 1.0)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# now let's look at some functions of the DataSet class

# DataSet objects can e.g. be ...
data_copy = data.copy()                                              # deepcopied
#data_copy.scale_range(data_range)                                # scaled
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
classification.plot(plot_class_sparsegrid=True, plot_class_combi_scheme=False, plot_class_dataset=False, plot_class_density_estimation=True)

# if we already added some testing data to the Classification object (which we did in the initialization process, 20% of samples are testing samples), we can print the current evaluation
classification.print_evaluation()

# we can also add more testing data and print the results immediately
#with_classes.set_name("Test_new_data")
#classification.test_data(with_classes, print_output=False)

# and we can call the Classification object to perform blind classification on a dataset with unknown class assignments to its samples
#data_copy.remove_labels(1.0)
classif_data = classification.get_data()
classif_data.remove_labels(1.0)
data_copy.remove_labels(1.0)
calcult_classes = classification(data_copy)

# because we used 2D datasets before, we can plot the results to easily see which samples were classified correctly and which not
correct_classes = data.copy()
correct_classes.scale_range(data_range)
correct_classes.set_name('Correct_Classes')
calcult_classes.set_name('Calculated_Classes')
retfig1 = calcult_classes.plot()
retfig0 = correct_classes.plot()

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
boundary = False
modified_basis = False
tolerance = -1.0
margin = 0.5
classification_dimwise.perform_classification_dimension_wise(masslumping=False, lambd=0.0, minimum_level=1, maximum_level=2,
                                                     reuse_old_values=True, numeric_calculation=False,
                                                     boundary=boundary, modified_basis=modified_basis, one_vs_others=one_vs_others,
                                                     tolerance=tolerance, margin=margin, max_evaluations=max_evals,
                                                             pre_scaled_data=True, filename='DimWiseClassification_',
                                                             error_calculator=error_calculator, rebalancing=True)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# now we can perform some other operations on this classification object


# we could e.g. plot its classificators and corresponding density estimations
classification_dimwise.plot(plot_class_sparsegrid=True, plot_class_combi_scheme=True, plot_class_dataset=False, plot_class_density_estimation=True)

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
retfig1 = calcult_classes_dimwise.plot()
retfig0 = correct_classes_dimwise.plot()


log_info('--- DimWiseClassification end ---')