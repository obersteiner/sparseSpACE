from sys import path

path.append('../src/')
import numpy as np
from ErrorCalculator import *
from GridOperation import *
from StandardCombi import *
from numpy.linalg import norm


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


# dimension of the problem
dim = 2

# define number of samples
size = 5000

# define boundaries
a = np.zeros(dim)
b = np.ones(dim)

# csv dataset file
data = "Datasets/Circles500.csv"
data_name = "circles"

# define lambda
lambd = 0.0

pointsPerDim = 3

minimum_level = 4
maximum_level = 5

X = np.linspace(0.25, 0.75, pointsPerDim)
Y = np.linspace(0.25, 0.75, pointsPerDim)
X, Y = np.meshgrid(X, Y)
points = list(map(lambda x, y: (x, y), X.flatten(), Y.flatten()))

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Full Grid
operation_fullgrid = DensityEstimation(data, dim, print_output=False, lambd=lambd)
combiObject_fullgrid = StandardCombi(a, b, operation=operation_fullgrid)

combiObject_fullgrid.perform_operation(maximum_level, maximum_level)
numb_points_fullgrid = operation_fullgrid.grid.get_num_points()
result_fullgrid = combiObject_fullgrid(points)
# result_fullgrid = result_fullgrid.reshape((pointsPerDim, pointsPerDim))

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Combi 1-max
operation_combi = DensityEstimation(data, dim, print_output=False, lambd=lambd)
combiObject_combi = StandardCombi(a, b, operation=operation_combi)

combiObject_combi.perform_operation(1, maximum_level)
numb_points_combi = numb_points_sparse_grid(combiObject_combi)
result_combi = combiObject_combi(points)
# result_combi = result_combi.reshape((pointsPerDim, pointsPerDim))
diff_combi = np.subtract(result_fullgrid, result_combi)

l1_norm_combi = norm(diff_combi, 1)
l2_norm_combi = norm(diff_combi, 2)
lmax_norm_combi = norm(diff_combi, np.inf)

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# TODO Best combi component


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Combi Lmin Lmax
operation_lmin_lmax = DensityEstimation(data, dim, print_output=False, lambd=lambd)
combiObject_lmin_lmax = StandardCombi(a, b, operation=operation_lmin_lmax)

combiObject_lmin_lmax.perform_operation(minimum_level, maximum_level)
numb_points_lmin_lmax = numb_points_sparse_grid(combiObject_lmin_lmax)
result_lmin_lmax = combiObject_lmin_lmax(points)
# result_lmin_lmax = result_lmin_lmax.reshape((pointsPerDim, pointsPerDim))
diff_lmin_lmax = np.subtract(result_fullgrid, result_lmin_lmax)

l1_norm_lmin_lmax = norm(diff_lmin_lmax, 1)
l2_norm_lmin_lmax = norm(diff_lmin_lmax, 2)
lmax_norm_lmin_lmax = norm(diff_lmin_lmax, np.inf)

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Mass lumping 1-max
operation_lumping = DensityEstimation(data, dim, masslumping=True, print_output=False, lambd=lambd)
combiObject_lumping = StandardCombi(a, b, operation=operation_lumping)

combiObject_lumping.perform_operation(1, maximum_level)
numb_points_lumping = numb_points_sparse_grid(combiObject_lumping)
result_lumping = combiObject_lumping(points)
# result_lumping = result_lumping.reshape((pointsPerDim, pointsPerDim))
diff_lumping = np.subtract(result_fullgrid, result_lumping)

l1_norm_lumping = norm(diff_lumping, 1)
l2_norm_lumping = norm(diff_lumping, 2)
lmax_norm_lumping = norm(diff_lumping, np.inf)

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Results
print("Fullgrid")
print("Result: ", result_fullgrid, " #Points: ", numb_points_fullgrid, " Lambda: ", lambd)
print("Combi: ")
print("L1: ", l1_norm_combi, " L2: ", l2_norm_combi, " Lmax: ", lmax_norm_combi, " #Points: ", numb_points_combi, " Lambda: ", lambd)

print("Lmin Lmax: ")
print("L1: ", l1_norm_lmin_lmax, " L2: ", l2_norm_lmin_lmax, " Lmax: ", lmax_norm_lmin_lmax, " #Points: ", numb_points_lmin_lmax, " Lambda: ", lambd)

print("Lumping: ")
print("L1: ", l1_norm_lumping, " L2: ", l2_norm_lumping, " Lmax: ", lmax_norm_lumping, " #Points: ", numb_points_lumping, " Lambda: ", lambd)
with open("Results/result_" + data_name + "_" + str(minimum_level) + "_" + str(maximum_level), "w") as file:
    file.write(
        data_name + "Config:\n\tMin: " + str(minimum_level) + "\n\tMax: " + str(maximum_level) + "\n\tDim: " + str(dim) + "\n\tSize: " + str(size) + "\n\tpointsPerDim: " + str(pointsPerDim) + "\n")
    file.write("\tPoints: " + str(points) + "\n")
    file.write("\nFullgrid " + str(maximum_level) + ":\n")
    file.write("\tResult: " + str(list(result_fullgrid.flatten())) + "\n\t#Points: " + str(numb_points_fullgrid) + "\n\tLambda: " + str(lambd) + "\n")
    file.write("\nCombi 1 " + str(maximum_level) + ":\n")
    file.write(
        "\tResult: " + str(list(result_combi.flatten())) + "\n\tDiff: " + str(list(diff_combi.flatten())) + "\n\tL1: " + str(l1_norm_combi) + "\n\tL2: " + str(l2_norm_combi) + "\n\tLmax: " + str(
            lmax_norm_combi) + "\n\t#Points: " + str(numb_points_combi) + "\n\tLambda: " + str(lambd) + "\n")

    file.write("\nLmin Lmax " + str(minimum_level) + " " + str(maximum_level) + ":\n")
    file.write("\tResult: " + str(list(result_lmin_lmax.flatten())) + "\n\tDiff: " + str(list(diff_lmin_lmax.flatten())) + "\n\tL1: " + str(l1_norm_lmin_lmax) + "\n\tL2: " + str(
        l2_norm_lmin_lmax) + "\n\tLmax: " + str(lmax_norm_lmin_lmax) + "\n\t#Points: " + str(numb_points_lmin_lmax) + "\n\tLambda: " + str(lambd) + "\n")

    file.write("\nLumping 1 " + str(maximum_level) + ":\n")
    file.write("\tResult: " + str(list(result_lumping.flatten())) + "\n\tDiff: " + str(list(diff_lumping.flatten())) + "\n\tL1: " + str(l1_norm_lumping) + "\n\tL2: " + str(
        l2_norm_lumping) + "\n\tLmax: " + str(lmax_norm_lumping) + "\n\t#Points: " + str(numb_points_lumping) + "\n\tLambda: " + str(lambd) + "\n")

    file.close()
