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
log_info('data size: ' + str(size))
log_info('data dimension: ' + str(dim))

# sklearn_dataset = do.datasets.make_circles(n_samples=size, noise=0.05)
# data_set_name = 'Circles'
sklearn_dataset = do.datasets.make_moons(n_samples=size, noise=0.2)
data_set_name = 'Two Moons'
# sklearn_dataset = do.datasets.make_moons(n_samples=size, noise=0.3)
# sklearn_dataset = do.datasets.make_classification(size, n_features=dim, n_redundant=0, n_clusters_per_class=1, n_informative=1, n_classes=2)
# sklearn_dataset = do.datasets.make_classification(size, n_features=dim, n_redundant=0, n_clusters_per_class=1, n_informative=2, n_classes=3)
# data_set_name = 'Random Classes'
# sklearn_dataset = do.datasets.make_blobs(n_samples=size, n_features=dim, centers=3)
# data_set_name = 'Blobs'
#sklearn_dataset = do.datasets.make_gaussian_quantiles(n_samples=size, n_features=dim, n_classes=3)
#data_set_name = 'Gaussian Quantiles'

#error_config = (ErrorCalculatorSingleDimVolumeGuided(), False)
#error_config = (ErrorCalculatorSingleDimMisclassificationGlobal(), True)
error_configs = [(ErrorCalculatorSingleDimVolumeGuided(), False), (ErrorCalculatorSingleDimMisclassificationGlobal(), True)]
for error_config in error_configs:
    if error_configs.index(error_config) == 0:
        path_std = 'figures/vol_std'
        path_dimwise = 'figures/vol_dim-wise'
        path_std_classif = 'figures/vol_calculated_classes_std'
        path_dimwise_classif = 'figures/vol_calculated_classes_dim-wise'
    elif error_configs.index(error_config) == 1:
        path_std = 'figures/OvO_std'
        path_dimwise = 'figures/OvO_dim-wise'
        path_std_classif = 'figures/OvO_calculated_classes_std'
        path_dimwise_classif = 'figures/OvO_calculated_classes_dim-wise'
    error_calculator = error_config[0]
    one_vs_others = error_config[1]

    log_info('one vs others: ' + str(one_vs_others))
    log_info('error_calculator: ' + str(type(error_calculator)))

    # now we can transform this dataset into a DataSet object and give it an appropriate name
    data = do.DataSet(sklearn_dataset, name=data_set_name)
    #data.plot()
    data_range = (0.0, 1.0)

    split_evenly = True

    data_copy = data.copy()                                              # deepcopied
    part0, part1 = data_copy.split_pieces(0.5)                           # split
    data_copy = part0.concatenate(part1)                                 # concatenated
    data_copy.set_name('2nd_Set')                                        # renamed
    data_copy.remove_labels(0.2)                                        # freed of some class assignments to samples
    without_classes, with_classes = data_copy.split_without_labels()    # seperated into samples with and without classes

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
    data.plot(filename='figures/data_plot')
    ## plot single classes
    # for i in range(len(lab)):
    #     print_copy = data.copy()
    #     for j in range(len(lab)):
    #         if j != i:
    #             to_remove = np.argwhere(print_copy[1] == lab[j]).flatten()
    #             print_copy.remove_samples(to_remove)
    #     print_copy.plot()

    classification = do.Classification(data, split_percentage=0.8, split_evenly=split_evenly)

    max_level = 4
    print('classification max_level', max_level)
    log_info('classification standardCombi max_level: ' + str(max_level))
    classification.perform_classification(masslumping=False,
                                          lambd=0.0,
                                          minimum_level=1,
                                          maximum_level=max_level,
                                          one_vs_others=one_vs_others,
                                          pre_scaled_data=True)

    classification.plot(plot_class_sparsegrid=True, plot_class_combi_scheme=True, plot_class_dataset=True, plot_class_density_estimation=True, file_path=path_std)

    classification.print_evaluation()

    classif_data = classification.get_data()
    classif_data.remove_labels(1.0)
    data_copy.remove_labels(1.0)
    calcult_classes = classification(data_copy)

    correct_classes = data.copy()
    correct_classes.scale_range(data_range)
    correct_classes.set_name('Correct_Classes')
    calcult_classes.set_name('Calculated Classes standard combi')
    retfig1 = calcult_classes.plot(filename=path_std_classif)
    retfig0 = correct_classes.plot(filename='figures/correct_classes_std')

    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################

    classification_dimwise = do.Classification(data, split_percentage=0.8, split_evenly=split_evenly)

    #max_evals = (((2**max_level) - 1) * dim)
    max_evals = ((2**max_level) - 1) * dim - (dim - 1) + (2**dim) * prev_level(max_level, dim)
    print('classification max_evaluations', max_evals)
    log_info('classification dimwise max_evaluations: ' + str(max_evals))

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

    classification_dimwise.plot(plot_class_sparsegrid=True, plot_class_combi_scheme=True, plot_class_dataset=True, plot_class_density_estimation=True, file_path=path_dimwise)

    classification_dimwise.print_evaluation()

    data_copy.remove_labels(1.0)
    classif_dimwise_data = classification.get_data()
    classif_dimwise_data.remove_labels(1.0)
    calcult_classes_dimwise = classification_dimwise(classif_dimwise_data)

    correct_classes_dimwise = data.copy()
    correct_classes_dimwise.scale_range(data_range)
    correct_classes_dimwise.set_name('Correct Classes')
    calcult_classes_dimwise.set_name('Calculated classes dimension wise')
    retfig1 = calcult_classes_dimwise.plot(filename=path_dimwise_classif)
    retfig0 = correct_classes_dimwise.plot(filename='figures/correct_classes_dim-wise')


    log_info('--- DimWiseClassification end ---', True)