from sys import path

path.append('../src/')
import numpy as np
import matplotlib
from ErrorCalculator import *
from GridOperation import *
from StandardCombi import *
from numpy.linalg import norm


def numb_points_sparse_grid(combiObject: StandardCombi, scheme=False) -> int:
    """
    This method calculates the number of points of the sparse grid
    :param combiObject:
    :return: number of points
    """
    numpoints = 0
    for component_grid in combiObject.scheme:
        pointsgrid = combiObject.get_num_points_component_grid(component_grid.levelvector, False)
        if scheme == False:
            numpoints += pointsgrid * int(component_grid.coefficient)
        else:
            numpoints += pointsgrid
    return numpoints


def plot_errors(numb_points, lvecs, l1_norm, l2_norm, lmax_norm, filename: str = None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # plt.plot(numb_points_3, l2_norm_3, "-ro", label="Level 3")
    # plt.plot(numb_points_4, l2_norm_4, "-bo", label="Level 4")
    # plt.plot(numb_points_5, l2_norm_5, "-go", label="Level 5")

    for i, x in enumerate(numb_points):
        ax.axvline(x=x, color="k", alpha=0.5, linestyle="--")
        ax.text(x , (l1_norm[i] - l2_norm[i]) / 2, lvecs[i], horizontalalignment='center', verticalalignment='center', rotation=90)

    ax.plot(numb_points, l1_norm, "-ro", label="$L_1$-norm")
    ax.plot(numb_points, l2_norm, "-bo", label="$L_2$-norm")
    ax.plot(numb_points, lmax_norm, "-go", label="$L_{\infty}$-norm")

    ax.legend(loc="upper right")
    ax.set_xlabel("Number of Grid Points")
    ax.set_ylabel("Error Norm")
    ax.set_yscale("log")
    # ax.set_yticks([1, 5, 10, 25, 50, 100, 150, 200, 300])
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def generate_figure_errors(data, dim: int = 2, plot_figures=True, filename: str = None, lambd: float = 0.0):
    # define boundaries
    a = np.zeros(dim)
    b = np.ones(dim)

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Full Grid
    points_full = []
    numb_points_full = []
    result_full = []

    for i in range(3, 6, 1):
        operation_fullgrid = DensityEstimation(data, dim, print_output=False, lambd=lambd)
        combiObject_fullgrid = StandardCombi(a, b, operation=operation_fullgrid, print_output=False)
        combiObject_fullgrid.perform_operation(i, i)
        points = operation_fullgrid.grid.getPoints()
        points_full.append(points)
        numb_points_full.append(operation_fullgrid.grid.get_num_points())
        result_full.append(combiObject_fullgrid(points))

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Level 3
    numb_points_3 = []
    numb_points_3_scheme = []
    result_3 = []
    diff_3 = []
    l1_norm_3 = []
    l2_norm_3 = []
    lmax_norm_3 = []
    lvec_3 = []
    lvec_3_grids = []
    for i in range(1, 3, 1):
        lvec_3.append((i, 3))
        operation_3 = DensityEstimation(data, dim, print_output=False, lambd=lambd)
        combiObject_3 = StandardCombi(a, b, operation=operation_3, print_output=False)
        combiObject_3.perform_operation(i, 3)
        lvec_3_grids.append(len(combiObject_3.scheme))
        numb_points_3.append(numb_points_sparse_grid(combiObject_3))
        numb_points_3_scheme.append(numb_points_sparse_grid(combiObject_3, True))
        result = combiObject_3(points_full[0])
        result_3.append(result)
        diff = np.subtract(result_full[0], result)
        diff_3.append(diff)
        l1_norm_3.append(norm(diff, 1))
        l2_norm_3.append(norm(diff, 2))
        lmax_norm_3.append(norm(diff, np.inf))

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Level 4
    numb_points_4 = []
    numb_points_4_scheme = []
    result_4 = []
    diff_4 = []
    l1_norm_4 = []
    l2_norm_4 = []
    lmax_norm_4 = []
    lvec_4 = []
    lvec_4_grids = []
    for i in range(1, 4, 1):
        lvec_4.append((i, 4))
        operation_4 = DensityEstimation(data, dim, print_output=False, lambd=lambd)
        combiObject_4 = StandardCombi(a, b, operation=operation_4, print_output=False)
        combiObject_4.perform_operation(i, 4)
        lvec_4_grids.append(len(combiObject_4.scheme))
        numb_points_4.append(numb_points_sparse_grid(combiObject_4))
        numb_points_4_scheme.append(numb_points_sparse_grid(combiObject_4, True))
        result = combiObject_4(points_full[1])
        result_4.append(result)
        diff = np.subtract(result_full[1], result)
        diff_4.append(diff)
        l1_norm_4.append(norm(diff, 1))
        l2_norm_4.append(norm(diff, 2))
        lmax_norm_4.append(norm(diff, np.inf))

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Level 5
    numb_points_5 = []
    numb_points_5_scheme = []
    result_5 = []
    diff_5 = []
    l1_norm_5 = []
    l2_norm_5 = []
    lmax_norm_5 = []
    lvec_5 = []
    lvec_5_grids = []
    for i in range(1, 5, 1):
        lvec_5.append((i, 5))
        operation_5 = DensityEstimation(data, dim, print_output=False, lambd=lambd)
        combiObject_5 = StandardCombi(a, b, operation=operation_5, print_output=False)
        combiObject_5.perform_operation(i, 5)
        lvec_5_grids.append(len(combiObject_5.scheme))
        numb_points_5.append(numb_points_sparse_grid(combiObject_5))
        numb_points_5_scheme.append(numb_points_sparse_grid(combiObject_5, True))
        result = combiObject_5(points_full[2])
        result_5.append(result)
        diff = np.subtract(result_full[2], result)
        diff_5.append(diff)
        l1_norm_5.append(norm(diff, 1))
        l2_norm_5.append(norm(diff, 2))
        lmax_norm_5.append(norm(diff, np.inf))

    # Print output
    if filename is not None:
        file = open(filename, "a")
    else:
        file = None

    print("Comparison Lmin - Lmax: \n", file=file)
    print("Level vectoren:\t", file=file)
    print("3: " + str(lvec_3) + "\t", file=file)
    print("4: " + str(lvec_4) + "\t", file=file)
    print("5: " + str(lvec_5) + "\n", file=file)

    print("Number of Points Scheme:\t")
    print("3: " + str(numb_points_3_scheme) + "\t", file=file)
    print("4: " + str(numb_points_4_scheme) + "\t", file=file)
    print("5: " + str(numb_points_5_scheme) + "\n", file=file)

    print("Number of Points Sparse Grid:\t", file=file)
    print("3: " + str(numb_points_3) + "\t", file=file)
    print("4: " + str(numb_points_4) + "\t", file=file)
    print("5: " + str(numb_points_5) + "\n", file=file)

    print("Number of Grids:\t")
    print("3: " + str(lvec_3_grids) + "\t", file=file)
    print("4: " + str(lvec_4_grids) + "\t", file=file)
    print("5: " + str(lvec_5_grids) + "\n", file=file)

    print("L1-norm:\t")
    print("3: " + str(l1_norm_3) + "\t", file=file)
    print("4: " + str(l1_norm_4) + "\t", file=file)
    print("5: " + str(l1_norm_5) + "\n", file=file)

    print("L2-norm:\t")
    print("3: " + str(l2_norm_3) + "\t", file=file)
    print("4: " + str(l2_norm_4) + "\t", file=file)
    print("5: " + str(l2_norm_5) + "\n", file=file)

    print("L_infinty-norm:\t")
    print("3: " + str(lmax_norm_3) + "\t", file=file)
    print("4: " + str(lmax_norm_4) + "\t", file=file)
    print("5: " + str(lmax_norm_5) + "\n", file=file)

    if filename is not None:
        file.close()

    if plot_figures:
        plot_errors(numb_points_3, lvec_3, l1_norm_3, l2_norm_3, lmax_norm_3, filename + "_3.png" if filename is not None else filename)
        plot_errors(numb_points_4, lvec_4, l1_norm_4, l2_norm_4, lmax_norm_4, filename + "_4.png" if filename is not None else filename)
        plot_errors(numb_points_5, lvec_5, l1_norm_5, l2_norm_5, lmax_norm_5, filename + "_5.png" if filename is not None else filename)
