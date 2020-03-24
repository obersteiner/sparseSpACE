from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from numpy.linalg import norm
from GridOperation import *
from StandardCombi import *
import dill


def numb_points_sparse_grid(combiObject: StandardCombi) -> int:
    """
    This method calculates the number of points of the sparse grid
    :param combiObject:
    :return: number of points
    """
    numpoints = 0
    for component_grid in combiObject.scheme:
        pointsgrid = combiObject.get_num_points_component_grid(component_grid.levelvector, False)

        numpoints += pointsgrid * int(component_grid.coefficient)
    return numpoints


def plot_gaussian(filename: str = None, dim: int = 2, minimum_level: int = 1, maximum_level: int = 5) -> None:
    # define boundaries
    a = np.zeros(dim)
    b = np.ones(dim)

    data = ""

    # Define the gaussian distribution
    mean = np.array([0.5] * dim)
    sigma = np.array([0.25] * dim)
    cov = np.diag(sigma ** 2)
    rv = multivariate_normal(mean, cov)

    operation = DensityEstimation(data, dim, print_output=False)
    combiObject = StandardCombi(a, b, operation=operation, print_output=False)
    combiObject.set_combi_parameters(minimum_level, maximum_level)

    points = np.unique(combiObject.get_points_and_weights()[0], axis=0)

    x, y = zip(*points)
    z = rv.pdf(points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def calculate_gaussian_values(data, values_sgpp, dim: int = 5, minimum_level: int = 1, maximum_level: int = 5, lambd: float = 0.0, filename: str = None):
    # define boundaries
    a = np.zeros(dim)
    b = np.ones(dim)

    # define probability density function
    mean = np.array([0.5] * dim)
    sigma = np.array([0.25] * dim)
    cov = np.diag(sigma ** 2)
    rv = multivariate_normal(mean, cov)

    # create grid operation and combi object
    operation = DensityEstimation(data, dim, print_output=False, lambd=lambd)
    combiObject_combi = StandardCombi(a, b, operation=operation, print_output=False)
    combiObject_combi.perform_operation(minimum_level, maximum_level)

    # combiObject_combi.set_combi_parameters(minimum_level, maximum_level)
    # operation.surpluses = dill.load(open("../../src/DE/surpluses_" + str(maximum_level) + "_" + str(lambd) + "_" + str(dim), "rb"))
    # dill.dump(operation.get_result(), open("../../src/DE/surpluses_" + str(maximum_level) + "_" + str(lambd) + "_" + str(dim), "wb"))

    # get the sparse grid points
    points = np.unique(combiObject_combi.get_points_and_weights()[0], axis=0)
    numb_points = numb_points_sparse_grid(combiObject_combi)

    # calculate the reference density and calculate the difference
    results_density = np.vstack(rv.pdf(points))
    result_combi = combiObject_combi(points)
    diff_combi = np.subtract(results_density, result_combi)

    # calculate the error norms
    l1_norm_combi = np.linalg.norm(diff_combi, 1)
    l2_norm_combi = np.linalg.norm(diff_combi, 2)
    lmax_norm_combi = np.linalg.norm(diff_combi, np.inf)

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # SG++ max
    # load result from .csv file
    result_sgpp = np.genfromtxt(values_sgpp, delimiter=',')
    result_sgpp = np.vstack(result_sgpp)

    # calculate difference and error norms
    diff_sgpp = np.subtract(results_density, result_sgpp)
    l1_norm_sgpp = np.linalg.norm(diff_sgpp, 1)
    l2_norm_sgpp = np.linalg.norm(diff_sgpp, 2)
    lmax_norm_sgpp = np.linalg.norm(diff_sgpp, np.inf)

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Mass lumping 1-max
    # create grid operation and combi object
    operation_lumping = DensityEstimation(data, dim, masslumping=True, print_output=False, lambd=lambd)
    combiObject_lumping = StandardCombi(a, b, operation=operation_lumping, print_output=False)
    combiObject_lumping.perform_operation(1, maximum_level)

    # combiObject_lumping.set_combi_parameters(minimum_level, maximum_level)
    # operation_lumping.surpluses = dill.load(open("../../src/DE/surpluses_lumping_" + str(maximum_level) + "_" + str(lambd) + "_" + str(dim), "rb"))
    # dill.dump(operation_lumping.get_result(), open("../../src/DE/surpluses_lumping_" + str(maximum_level) + "_" + str(lambd) + "_" + str(dim), "wb"))

    # calculate difference to the actual density
    result_lumping = combiObject_lumping(points)
    diff_lumping = np.subtract(results_density, result_lumping)

    # calculate error norms
    l1_norm_lumping = np.linalg.norm(diff_lumping, 1)
    l2_norm_lumping = np.linalg.norm(diff_lumping, 2)
    lmax_norm_lumping = np.linalg.norm(diff_lumping, np.inf)

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Results
    if filename is not None:
        file = open(filename + ".txt", "a")
    else:
        file = None

    # Print output
    print("Config:\n\tMin: 1" + "\n\tMax: " + str(maximum_level) + "\n\tDim: " + str(dim) + "\tSize: " + str(len(operation.data)), file=file)
    print("\tPoints: " + str(points.tolist()), file=file)
    print("\tNumber of sample points: " + str(numb_points) + "\n\tLambda: " + str(lambd), file=file)
    print("Density Function " + str(maximum_level) + ":", file=file)
    print("\tResult: " + str(list(results_density.flatten())), file=file)
    print("\t#Points: " + str(numb_points), file=file)

    print("SG++ " + str(maximum_level) + ":", file=file)
    print("\tResult: " + str(list(result_sgpp.flatten())) + "\n\tDiff: " + str(list(diff_sgpp.flatten())), file=file)
    print("\tL1: " + str(l1_norm_sgpp) + "\n\tL2: " + str(l2_norm_sgpp) + "\tLmax: " + str(lmax_norm_sgpp), file=file)
    print("\t#Points: " + str(numb_points), file=file)

    print("Combi 1 " + str(maximum_level) + ":", file=file)
    print("\tResult: " + str(list(result_combi.flatten())) + "\n\tDiff: " + str(list(diff_combi.flatten())), file=file)
    print("\tL1: " + str(l1_norm_combi) + "\n\tL2: " + str(l2_norm_combi) + "\tLmax: " + str(lmax_norm_combi), file=file)
    print("\t#Points: " + str(numb_points), file=file)

    print("Lumping 1 " + str(maximum_level) + ":", file=file)
    print("\tResult: " + str(list(result_lumping.flatten())) + "\n\tDiff: " + str(list(diff_lumping.flatten())), file=file)
    print("\tL1: " + str(l1_norm_lumping) + "\n\tL2: " + str(l2_norm_lumping) + "\tLmax: " + str(lmax_norm_lumping), file=file)
    print("\t#Points: " + str(numb_points) + "\n", file=file)

    if filename is not None:
        file.close()
