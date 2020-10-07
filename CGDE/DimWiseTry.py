#%matplotlib inline
from sys import path
path.append('../src/')
path.append('../SGDE')
path.append('../SGDE/Datasets')

import numpy as np
import scipy as sp
import math

from spatiallyAdaptiveSingleDimension2 import *
from Function import *
from ErrorCalculator import *

# sgde tut
from GridOperation import *
from StandardCombi import *
from sklearn import datasets


from sklearn.neighbors import KernelDensity

import cProfile
import pstats

def prev_level(l, d):
    if l - 2 <= 0:
        return 1
    else:
        return (2**(l-2) - 1) * d + prev_level(l-2, d)

def points(l, d):
    res = 0
    for k in range(0, d):
        res += ((-1)**k) * binomial(d - 1, k) + ((2**l) + (d - 1) - k)


def binomial(n, k):
    return math.factorial(n - 1) / (math.factorial(k) * math.factorial(n - k))


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

# path to where the figures should be saved
fig_path = 'figures/'
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
minimum_level, maximum_level = 1, 5
print('max level of standard combirid:', minimum_level, ' : ', maximum_level)
# define starting level of dimension wise combigrid
lmin, lmax = 1, 3
print('lim/lmax of dimWise grid: ', lmin, ' : ', lmax)
# error tolerance
tolerance = 0.01
print('error tolerance:', tolerance)
# error margin
margin = 0.5
print('error margin: ', margin)
# maximum amount of new grid_points
#max_evaluations = 256
max_levels = maximum_level
max_evaluations = ((2 ** max_levels) - 1) * dim - (dim - 1) + (2 ** dim) * prev_level(max_levels, dim)
print('max evaluations for dimWise:', max_evaluations)
# plot the resulting combi-scheme with each refinement
do_plot = False
print('refinement plotting:', do_plot)
print_output = False

# kde parameters

kde_bandwidths = [0.01, 0.05, 0.1, 0.25, 0.5]
print('kde_bandwidth:', kde_bandwidths)

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
    #mean = np.array([0.0] * dim)
    #sigma = np.array([0.25]*dim)
    #cov = np.diag(sigma**2)
    #data = np.random.multivariate_normal(mean, cov, size)
    mean_a = np.array([-0.5] * dim)
    mean_b = np.array([+0.5] * dim)
    sigma = np.array([0.25] * dim)
    cov = np.diag(sigma ** 2)
    class_a = np.random.multivariate_normal(mean_a, cov, int(size / 2))
    class_b = np.random.multivariate_normal(mean_b, cov, int(size / 2))
    data = np.vstack((class_a, class_b))

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
plot_dataset(data, dim, 'figures/dataPlot_'+data_set)

########### GRID EVALUATIONS

### Standard Combi
#for i in range(max(minimum_level+1, maximum_level-2), maximum_level+1):
for i in list(range(3, max_levels+1)):
    maximum_level = i
    # define operation to be performed
    operation = DensityEstimation(data, dim, lambd=lambd, reuse_old_values=reuse_old_values, classes=class_signs, print_output=print_output)

    # create the combiObject and initialize it with the operation
    combiObject = StandardCombi(a, b, operation=operation, print_output=print_output)

    if do_plot:
        print("Plot of dataset:")
        operation.plot_dataset(filename=fig_path+'stdCombi_'+data_set+'_dataSet_')
    # perform the density estimation operation, has to be done before the printing and plotting
    combiObject.perform_operation(minimum_level, maximum_level)
    #cProfile.run(
    #    'combiObject.perform_operation(minimum_level, maximum_level)',
    #    filename='StandardCombiProfile.txt')
    #std_p_stat = pstats.Stats('StandardCombiProfile.txt')
    if do_plot:
        print("Combination Scheme:")
        # when you pass the operation the function also plots the contour plot of each component grid
        combiObject.print_resulting_combi_scheme(filename=fig_path+'stdCombi_'+data_set+'_scheme_'+'lmax-'+str(maximum_level), operation=operation)
    if do_plot:
        print("Sparse Grid:")
        combiObject.print_resulting_sparsegrid(filename=fig_path+'stdCombi_'+data_set+'_grid'+'lmax-'+str(maximum_level), markersize=20)
    if do_plot:
        print("Plot of density estimation")
        # when contour = True, the contour plot is shown next to the 3D plot
        combiObject.plot(filename=fig_path+'stdCombi_'+data_set+'_contour_'+'lmax-'+str(maximum_level), contour=True)

################## dimension wise
#############
timings = {}  # pass this dict to the operation and grid scheme to collect execution time information

newGrid = GlobalTrapezoidalGrid(a=np.zeros(dim), b=np.ones(dim), modified_basis=modified_basis, boundary=boundary)

if 'class' in data_set:
    # errorOperator = ErrorCalculatorSingleDimMisclassification()
    errorOperator = ErrorCalculatorSingleDimMisclassificationGlobal()
else:
    errorOperator = ErrorCalculatorSingleDimVolumeGuided()


# define operation to be performed
op = DensityEstimation(data, dim, grid=newGrid, lambd=lambd, classes=class_signs, reuse_old_values=reuse_old_values, numeric_calculation=numeric_calculation, print_output=print_output)
# create the combiObject and initialize it with the operation
SASD = SpatiallyAdaptiveSingleDimensions2(a, b, operation=op, margin=margin, timings=timings, rebalancing=False)
if do_plot:
    print("Plot of dataset:")
    op.plot_dataset(filename=fig_path+'dimWise_'+data_set+'_dataSet')
# perform the density estimation operation, has to be done before the printing and plotting
SASD.performSpatiallyAdaptiv(lmin, lmax, errorOperator, tolerance, max_evaluations=max_evaluations, do_plot=do_plot)
# if print_output:
#     cProfile.run('SASD.performSpatiallyAdaptiv(lmin, lmax, errorOperator, tolerance, max_evaluations=max_evaluations, do_plot=do_plot)',
#              filename='DimWiseAdaptivProfile.txt')
# else:
#     cProfile.run('SASD.performSpatiallyAdaptiv(lmin, lmax, errorOperator, tolerance, max_evaluations=max_evaluations, do_plot=do_plot, print_output=False)',
#              filename='DimWiseAdaptivProfile.txt')
# dimWise_p_stat = pstats.Stats('DimWiseAdaptivProfile.txt')



# print the execution times
for k in timings.keys():
    print(k, timings[k])

if reuse_old_values and op.debug:
    print('reuse abs_diffs: ', op.reuse_abs_diff)
    print('reuse avg abs_diffs: ', sum([x[1] for x in op.reuse_abs_diff]) / len(op.reuse_abs_diff))
    print('reuse abs avg abs_diffs: ', sum([abs(x[1]) for x in op.reuse_abs_diff]) / len(op.reuse_abs_diff))
    print('reuse max min abs_diffs: ', max([x[1] for x in op.reuse_abs_diff]), min([x[1] for x in op.reuse_abs_diff]))
    print('reuse rel_diffs: ', op.reuse_rel_diff)
    print('reuse avg rel_diffs: ', sum([x[1] for x in op.reuse_rel_diff]) / len(op.reuse_rel_diff))
    print('reuse abs avg rel_diffs: ', sum([abs(x[1]) for x in op.reuse_rel_diff]) / len(op.reuse_rel_diff))
    print('reuse max min rel_diffs: ', max([x[1] for x in op.reuse_rel_diff]), min([x[1] for x in op.reuse_rel_diff]))

#op.post_processing()
if do_plot:
    print("Combination Scheme:")
    # when you pass the operation the function also plots the contour plot of each component grid
    SASD.print_resulting_combi_scheme(filename=fig_path+'dimWise_'+data_set+'_scheme', operation=op)
if do_plot:
    print("Sparse Grid:")
    SASD.print_resulting_sparsegrid(filename=fig_path+'dimWise_'+data_set+'_grid', markersize=20)
if do_plot:
    print("Plot of density estimation")
    # when contour = True, the contour plot is shown next to the 3D plot
    SASD.plot(filename=fig_path+'dimWise_'+data_set+'_contour', contour=True)

SASD.min_max_scale_surplusses()

################### Kernel Density Estimation
kdes = None
if data_set == 'floats':
    print('')
elif data_set == 'std exponential':

    kdes = [KernelDensity(kernel='exponential', bandwidth=x).fit(data) for x in kde_bandwidths]
elif data_set == 'std normal':
    kdes = [KernelDensity(kernel='gaussian', bandwidth=x).fit(data) for x in kde_bandwidths]
elif data_set == 'multi normal':
    kdes = [KernelDensity(kernel='gaussian', bandwidth=x).fit(data) for x in kde_bandwidths]
elif data_set == 'uniform':
    kdes = [KernelDensity(kernel='gaussian', bandwidth=x).fit(data) for x in kde_bandwidths]
elif data_set == 'line':
    kdes = [KernelDensity(kernel='linear', bandwidth=x).fit(data) for x in kde_bandwidths]
elif data_set == 'cross':
    kdes = [KernelDensity(kernel='linear', bandwidth=x).fit(data) for x in kde_bandwidths]
elif data_set == 'moon':
    kdes = [KernelDensity(kernel='gaussian', bandwidth=x).fit(data) for x in kde_bandwidths]
elif data_set == 'circle':
    kdes = [KernelDensity(kernel='gaussian', bandwidth=x).fit(data) for x in kde_bandwidths]
if kdes and do_plot:
    for kde_i, kde in enumerate(kdes):
        print('Plot of KDE result:')
        plotdimension = 0
        fontsize = 30
        plt.rcParams.update({'font.size': fontsize})
        xArray = np.linspace(a[0], b[0], 10 ** 2)
        yArray = np.linspace(a[1], b[1], 10 ** 2)
        XX = [x for x in xArray]
        YY = [y for y in yArray]
        points = list(get_cross_product([XX, YY]))

        X, Y = np.meshgrid(XX, YY, indexing="ij")
        Z = np.zeros(np.shape(X))

        f_values = np.asarray((kde.score_samples(points)))
        for i in range(len(X)):
            for j in range(len(X[i])):
                # print(X[i,j],Y[i,j],self.eval((X[i,j],Y[i,j])))
                Z[i, j] = f_values[j + i * len(X)] if f_values[j + i * len(X)] >= 0.0 else 0
        fig = plt.figure(figsize=(20, 10))

        # `ax` is a 3D-aware axis instance, because of the projection='3d' keyword argument to add_subplot
        ax = fig.add_subplot(1, 2, 1, projection='3d')

        ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        ax = fig.add_subplot(1, 2, 2)
        # TODO why do I have to transpose here so it plots in the right orientation?
        p = ax.imshow(np.transpose(Z), extent=[0.0, 1.0, 0.0, 1.0], origin='lower', cmap=cm.coolwarm)
        # ax.axis(aspect="image")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(p, cax=cax)

        plt.savefig('figures/kde_result_bdWdth-'+str(kde_bandwidths[kde_i]).replace('.','p'), bbox_inches='tight')
        #plt.show()
        # reset fontsize to default so it does not affect other figures
        #plt.rcParams.update({'font.size': plt.rcParamsDefault.get('font.size')})

## comparisons
# possible divergences to use for measuring how close the densities are https://en.wikipedia.org/wiki/F-divergence ; Kullbeck-Leibler and Pearsson
# besides the accuracy measure (i.e. Lp norm of the difference)
ref_vals = None
kde_vals = None
grid_vals = None
if data_set == 'floats':
    # define data (https://docs.scipy.org/doc/numpy-1.14.0/reference/routines.random.html)
    # random floats
    test_data = np.random.random((size, dim))
    test_data = scale_data(test_data, dim, scale)
    kde_vals = [np.array(kdes.score_samples(test_data)) for kde in kdes]
    grid_vals = SASD(test_data).reshape(size)
elif data_set == 'std exponential':
    # samples from the standard exponential distribution.
    test_data = np.random.standard_exponential((size, dim))
    test_data = scale_data(test_data, dim, scale)
    kde_vals = [np.array(kde.score_samples(test_data)) for kde in kdes]
    grid_vals = SASD(test_data).reshape(size)
elif data_set == 'std normal':
    # samples from the standard normal distribution
    test_data = np.random.standard_normal((size, dim))
    test_data = scale_data(test_data, dim, scale)
    ref_vals = np.array([sp.stats.norm().pdf(p) for p in test_data])
    kde_vals = [np.array(kde.score_samples(test_data)) for kde in kdes]
    grid_vals = SASD(test_data).reshape(size)
elif data_set == 'multi normal':
    # multivariate normal distribution
    #mean = np.array([0.0] * dim)
    #sigma = np.array([0.25]*dim)
    #cov = np.diag(sigma**2)
    #test_data = np.random.multivariate_normal(mean, cov, size)
    #test_data = scale_data(test_data, dim, scale)
    mean_a = np.array([-0.5] * dim)
    mean_b = np.array([+0.5] * dim)
    sigma = np.array([0.25] * dim)
    cov = np.diag(sigma ** 2)
    class_a = np.random.multivariate_normal(mean_a, cov, int(size / 2))
    class_b = np.random.multivariate_normal(mean_b, cov, int(size / 2))
    test_data = np.vstack((class_a, class_b))
    test_data = scale_data(test_data, dim, scale)

    #ref_vals = np.array([sp.stats.norm(cov, sigma).pdf(p)[0] * sp.stats.norm(cov, sigma).pdf(p)[1] for p in test_data])
    kde_vals = [np.array(kde.score_samples(test_data)) for kde in kdes]
    grid_vals = SASD(test_data).reshape(size)
elif data_set == 'uniform':
    # uniform distribution
    test_data = np.random.uniform(0.0, 1.0, [size, 2])
    test_data = scale_data(test_data, dim, scale)
    ref_vals = np.array([sp.stats.uniform(0.0, 1.0).pdf(p[0]) * sp.stats.uniform(0.0, 1.0).pdf(p[1]) for p in test_data])
    kde_vals = [np.array(kde.score_samples(test_data)) for kde in kdes]
    grid_vals = SASD(test_data).reshape(size)
elif data_set == 'line':
    # Line
    uni = np.random.uniform(0.0, 1.0, size)
    constant = np.ones(size) * 0.5
    #constant = np.random.uniform(0.4, 0.6, size)
    test_data = np.vstack((constant, uni)).T
    ref_vals = np.array([1.0 * sp.stats.uniform(0.0, 1.0).pdf(p[0]) for p in test_data])
    kde_vals = [np.array(kde.score_samples(test_data)) for kde in kdes]
    #grid_vals = np.array([SASD(p)[0][0] for p in test_data])
    grid_vals = SASD(test_data).reshape(size)
elif data_set == 'cross':
    # Cross
    uni = np.random.uniform(0.0, 1.0, int(size / 2))
    #constant = np.ones(int(size / 2)) * 0.5
    constant = np.random.uniform(cross_const_dom[0], cross_const_dom[1], int(size / 2))
    test_data = np.vstack((np.hstack((constant, uni)), np.hstack((uni, constant)))).T
    test_data = scale_data(test_data, dim, scale)
    #ref_vals = np.array([sp.stats.uniform(0.0, 1.0).pdf(p[0]) * sp.stats.uniform(0.0, 1.0).pdf(p[1]) for p in test_data])

    dom_diff = cross_const_dom[1] - cross_const_dom[0]
    ref_1 = np.array([sp.stats.uniform(0.0, dom_diff).pdf(test_data[i][0]) * sp.stats.uniform(0.0, 1.0).pdf(test_data[i][1]) for i in range(0, int(len(test_data)/2))])
    ref_2 = np.array([sp.stats.uniform(0.0, dom_diff).pdf(test_data[i][1]) * sp.stats.uniform(0.0, 1.0).pdf(test_data[i][0]) for i in range(int(len(test_data)/2), len(test_data))])
    ref_vals = np.hstack((ref_1, ref_2))

    kde_vals = [np.array(kde.score_samples(test_data)) for kde in kdes]
    grid_vals = SASD(test_data).reshape(size)
elif data_set == 'moon':
    # scikit learn datasets
    test_data = datasets.make_moons(size, noise=0.1)[0]
    test_data = scale_data(test_data, dim, scale)
    kde_vals = [np.array(kde.score_samples(test_data)) for kde in kdes]
    grid_vals = SASD(test_data).reshape(size)
elif data_set == 'circle':
    test_data = datasets.make_circles(size, noise=0.1)[0]
    test_data = scale_data(test_data, dim, scale)
    kde_vals = [np.array(kde.score_samples(test_data)) for kde in kdes]
    grid_vals = SASD(test_data).reshape(size)

if ref_vals is not None:
    print('avg sample diff ground - grid:', np.linalg.norm(ref_vals - grid_vals, 2) / size)
    for vals in kde_vals:
        print('avg sample diff ground - kde:', np.linalg.norm(ref_vals - vals, 2) / size)
if kde_vals is not None:
    for vals in kde_vals:
        print('avg sample diff kde - grid:', np.linalg.norm(vals - grid_vals, 2) / size)

if ref_vals is not None:
    print('KL_grid - ground:', scipy.stats.entropy(grid_vals, ref_vals)) # Kullbeck Leibler
    for vals in kde_vals:
        print('KL_kde - ground:', scipy.stats.entropy(vals, ref_vals))
if kde_vals is not None:
    for vals in kde_vals:
        print('KL_grid - kde:', scipy.stats.entropy(grid_vals, vals))

if ref_vals is not None:
    print('Pearsson corr ground - grid:', scipy.stats.pearsonr(ref_vals, grid_vals))
    for vals in kde_vals:
        print('Pearsson corr ground - kde:', scipy.stats.pearsonr(ref_vals, vals))
if kde_vals is not None:
    for vals in kde_vals:
        print('Pearsson corr kde - grid:', scipy.stats.pearsonr(grid_vals, vals))


# print("############# Standard Combi Profiling #############")
# std_p_stat.sort_stats(pstats.SortKey.TIME).print_stats(100)
# std_p_stat.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(100)
# print("############# DimWise Adaptive Profiling #############")
# dimWise_p_stat.sort_stats(pstats.SortKey.TIME).print_stats(100)
# dimWise_p_stat.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(100)