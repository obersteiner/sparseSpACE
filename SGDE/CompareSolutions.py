from sys import path

path.append('../src/')
import numpy as np
from GridOperation import *
from StandardCombi import *
from numpy.linalg import norm
from typing import Sequence


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


def plot_L2_error(difference: Sequence[float], points: Sequence[Sequence[float]], filename: str = None, threshold: int = 0.1):
    """
    This method plots the scatter plot for
    :param difference:
    :param points:
    :param filename:
    :param threshold:
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    X, Y = zip(*points)
    Z = difference ** 2
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)

    greater_than_threshold = [i for i, val in enumerate(Z) if val > threshold]
    p = ax.scatter(X, Y, Z)
    p = ax.scatter(X[greater_than_threshold], Y[greater_than_threshold], Z[greater_than_threshold], color="g")
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()
    return p


def calculate_comparison(data, values_sgpp, dim: int = 2, maximum_level: int = 5, lambd: float = 0.0, filename: str = None):
    # define boundaries
    a = np.zeros(dim)
    b = np.ones(dim)

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Full Grid
    operation_fullgrid = DensityEstimation(data, dim, print_output=False, lambd=lambd)
    combiObject_fullgrid = StandardCombi(a, b, operation=operation_fullgrid, print_output=False)

    combiObject_fullgrid.perform_operation(maximum_level, maximum_level)

    points = operation_fullgrid.grid.getPoints()

    numb_points_fullgrid = operation_fullgrid.grid.get_num_points()
    result_fullgrid = combiObject_fullgrid(points)

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # SG++ max
    result_sgpp = np.genfromtxt(values_sgpp, delimiter=',')
    result_sgpp = np.vstack(result_sgpp)
    diff_sgpp = np.subtract(result_fullgrid, result_sgpp)
    l1_norm_sgpp = norm(diff_sgpp, 1)
    l2_norm_sgpp = norm(diff_sgpp, 2)
    lmax_norm_sgpp = norm(diff_sgpp, np.inf)

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Combination technique 1-max
    operation_combi = DensityEstimation(data, dim, print_output=False, lambd=lambd)
    combiObject_combi = StandardCombi(a, b, operation=operation_combi, print_output=False)

    combiObject_combi.perform_operation(1, maximum_level)
    numb_points_combi = numb_points_sparse_grid(combiObject_combi)
    result_combi = combiObject_combi(points)

    diff_combi = np.subtract(result_fullgrid, result_combi)

    l1_norm_combi = norm(diff_combi, 1)
    l2_norm_combi = norm(diff_combi, 2)
    lmax_norm_combi = norm(diff_combi, np.inf)

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Mass lumping 1-max
    operation_lumping = DensityEstimation(data, dim, masslumping=True, print_output=False, lambd=lambd)
    combiObject_lumping = StandardCombi(a, b, operation=operation_lumping, print_output=False)

    combiObject_lumping.perform_operation(1, maximum_level)
    numb_points_lumping = numb_points_sparse_grid(combiObject_lumping)
    result_lumping = combiObject_lumping(points)
    diff_lumping = np.subtract(result_fullgrid, result_lumping)

    l1_norm_lumping = norm(diff_lumping, 1)
    l2_norm_lumping = norm(diff_lumping, 2)
    lmax_norm_lumping = norm(diff_lumping, np.inf)

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Best component grid
    components = []
    components_points = []
    result_components = []
    diff_components = []
    l1_norm_components = []
    l2_norm_components = []
    lmax_norm_components = []
    for component_grid in combiObject_combi.scheme:  # iterate over component grids
        components.append(component_grid.levelvector)
        result = combiObject_combi.interpolate_points(points, component_grid)
        components_points.append((component_grid.levelvector, combiObject_combi.get_num_points_component_grid(component_grid.levelvector, False)))
        result_components.append((component_grid.levelvector, combiObject_combi.get_num_points_component_grid(component_grid.levelvector, False), result))
        diff = np.subtract(result_fullgrid, result)
        diff_components.append(diff)
        l1_norm_components.append(norm(diff, 1))
        l2_norm_components.append(norm(diff, 2))
        lmax_norm_components.append(norm(diff, np.inf))

    best_component_l1 = (components[l1_norm_components.index(min(l1_norm_components))].tolist(), min(l1_norm_components))
    best_component_l2 = (components[l2_norm_components.index(min(l2_norm_components))].tolist(), min(l2_norm_components))
    best_component_lmax = (components[lmax_norm_components.index(min(lmax_norm_components))].tolist(), min(lmax_norm_components))

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Print results
    if filename is not None:
        file = open(filename, "a")
    else:
        file = None

    print("Config:\n\tMin: 1" + "\n\tMax: " + str(maximum_level) + "\n\tDim: " + str(dim) + "\n\tSize: " + str(len(operation_fullgrid.data)), file=file)
    print("\tPoints: " + str(points), file=file)
    print("\tNumber of sample points: " + str(numb_points_fullgrid) + "\n\tLambda: " + str(lambd), file=file)
    print("Fullgrid " + str(maximum_level) + ":", file=file)
    print("\tResult: " + str(list(result_fullgrid.flatten())) + "\n\t#Points: " + str(numb_points_fullgrid), file=file)

    print("SG++ " + str(maximum_level) + ":", file=file)
    print("\tResult: " + str(list(result_sgpp.flatten())) + "\tDiff: " + str(list(diff_sgpp.flatten())), file=file)
    print("\tL1: " + str(l1_norm_sgpp) + "\n\tL2: " + str(l2_norm_sgpp) + "\n\tLmax: " + str(lmax_norm_sgpp), file=file)
    print("\t#Points: " + str(numb_points_combi), file=file)

    print("Combi 1 " + str(maximum_level) + ":", file=file)
    print("\tResult: " + str(list(result_combi.flatten())) + "\tDiff: " + str(list(diff_combi.flatten())), file=file)
    print("\tL1: " + str(l1_norm_combi) + "\n\tL2: " + str(l2_norm_combi) + "\n\tLmax: " + str(lmax_norm_combi), file=file)
    print("\t#Points: " + str(numb_points_combi), file=file)

    print("Lumping 1 " + str(maximum_level) + ":", file=file)
    print("\tResult: " + str(list(result_lumping.flatten())) + "\tDiff: " + str(list(diff_lumping.flatten())), file=file)
    print("\tL1: " + str(l1_norm_lumping) + "\n\tL2: " + str(l2_norm_lumping) + "\n\tLmax: " + str(lmax_norm_lumping), file=file)
    print("\t#Points: " + str(numb_points_lumping), file=file)

    print("Best component grid 1 " + str(maximum_level) + ":", file=file)
    print("\tLevels and #Points: " + str(components_points), file=file)
    print("\tL1: " + str(l1_norm_components) + "\n\tBest component L1: " + str(best_component_l1), file=file)
    print("\tL2: " + str(l2_norm_components) + "\n\tBest component L2: " + str(best_component_l2), file=file)
    print("\tLmax: " + str(lmax_norm_components) + "\n\tBest component Lmax: " + str(best_component_lmax), file=file)
    print("\t#Points: " + str(numb_points_combi) + "\n\t#Grids: " + str(len(result_components))+"\n", file=file)

    if filename is not None:
        file.close()
