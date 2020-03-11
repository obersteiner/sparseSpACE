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
size = 500

# define boundaries
a = np.zeros(dim)
b = np.ones(dim)

# multivariate normal distribution
# mean = np.array([0.0] * dim)
# sigma = np.array([0.25]*dim)
# cov = np.diag(sigma**2)
# data = np.random.multivariate_normal(mean, cov, size)

# csv dataset file
data = "Datasets/moons.csv"
data_name = "moons"

# define lambda
lambd = 0.0

minimum_level = 2
maximum_level = 3

values_sgpp = "Results/SG++/sgpp_results_" + data_name + "_level_" + str(maximum_level) + "_dim_" + str(dim) + "_lambda_" + str(lambd) + ".csv"

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Full Grid
operation_fullgrid = DensityEstimation(data, dim, print_output=False, lambd=lambd)
combiObject_fullgrid = StandardCombi(a, b, operation=operation_fullgrid)

combiObject_fullgrid.perform_operation(maximum_level, maximum_level)

points = operation_fullgrid.grid.getPoints()
# operation_points = DensityEstimation(points, dim)
# operation_points.plot_dataset()

numb_points_fullgrid = operation_fullgrid.grid.get_num_points()
result_fullgrid = combiObject_fullgrid(points)
# result_fullgrid = result_fullgrid.reshape((pointsPerDim, pointsPerDim))
surplus_l2_norm_fullgrid = norm(np.concatenate(list(operation_fullgrid.get_result().values())), 2)
surplus_lmax_norm_fullgrid = norm(np.concatenate(list(operation_fullgrid.get_result().values())), np.inf)

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SG++ max
result_sgpp = np.genfromtxt(values_sgpp, delimiter=',')
result_sgpp = np.vstack(result_sgpp)
diff_sgpp = np.subtract(result_fullgrid, result_sgpp)
l1_norm_sgpp = norm(diff_sgpp, 1)
l2_norm_sgpp = norm(diff_sgpp, 2)
lmax_norm_sgpp = norm(diff_sgpp, np.inf)

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Combi 1-max
operation_combi = DensityEstimation(data, dim, print_output=False, lambd=lambd)
combiObject_combi = StandardCombi(a, b, operation=operation_combi)

combiObject_combi.perform_operation(1, maximum_level)
numb_points_combi = numb_points_sparse_grid(combiObject_combi)
result_combi = combiObject_combi(points)
# result_combi = result_combi.reshape((pointsPerDim, pointsPerDim))
diff_combi = np.subtract(result_fullgrid, result_combi)

surplus_l2_norm_combi = norm(np.concatenate(list(operation_combi.get_result().values())), 2)
surplus_lmax_norm_combi = norm(np.concatenate(list(operation_combi.get_result().values())), np.inf)

l1_norm_combi = norm(diff_combi, 1)
l2_norm_combi = norm(diff_combi, 2)
lmax_norm_combi = norm(diff_combi, np.inf)

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# best combi
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
save_diff_components = diff_components

# print_combi(combiObject_combi, save_diff_components, "Results/combi_error_" + data_name + "_" + str(minimum_level) + "_" + str(maximum_level) + "_" + str(lambd) + ".png")

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# TODO Best combi component
# TODO lowest error
# TODO error per grid as heatmap
# TODO use high dimensional gaussian
# TODO Table with different levels and one lambda
# TODO compare lmin lmax with different combinations of lmin lmax and cut numb of points from table
# TODO cut lmin lmax from table and make seperate table with number of points
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Combi Lmin Lmax
operation_lmin_lmax = DensityEstimation(data, dim, print_output=False, lambd=lambd)
combiObject_lmin_lmax = StandardCombi(a, b, operation=operation_lmin_lmax)

combiObject_lmin_lmax.perform_operation(minimum_level, maximum_level)
numb_points_lmin_lmax = numb_points_sparse_grid(combiObject_lmin_lmax)
result_lmin_lmax = combiObject_lmin_lmax(points)
# result_lmin_lmax = result_lmin_lmax.reshape((pointsPerDim, pointsPerDim))
diff_lmin_lmax = np.subtract(result_fullgrid, result_lmin_lmax)

surplus_l2_norm_lmin_lmax = norm(np.concatenate(list(operation_lmin_lmax.get_result().values())), 2)
surplus_lmax_norm_lmin_lmax = norm(np.concatenate(list(operation_lmin_lmax.get_result().values())), np.inf)

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

surplus_l2_norm_lumping = norm(np.concatenate(list(operation_lumping.get_result().values())), 2)
surplus_lmax_norm_lumping = norm(np.concatenate(list(operation_lumping.get_result().values())), np.inf)

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
with open("Results/result_" + data_name + "_" + str(minimum_level) + "_" + str(maximum_level) + "_" + str(lambd), "w") as file:
    file.write(
        data_name + "Config:\n\tMin: " + str(minimum_level) + "\n\tMax: " + str(maximum_level) + "\n\tDim: " + str(dim) + "\n\tSize: " + str(size) + "\n")
    file.write("\tPoints: " + str(points) + "\n")
    file.write("\tNumber of sample points: " + str(numb_points_fullgrid) + "\n")
    file.write("\nFullgrid " + str(maximum_level) + ":\n")
    file.write("\tResult: " + str(list(result_fullgrid.flatten())) + "\n\t#Points: " + str(numb_points_fullgrid) + "\n\tLambda: " + str(lambd) + "\n")

    file.write("\nSG++ " + str(maximum_level) + ":\n")
    file.write(
        "\tResult: " + str(list(result_sgpp.flatten())) + "\n\tDiff: " + str(list(diff_sgpp.flatten())) + "\n\tL1: " + str(l1_norm_sgpp) + "\n\tL2: " + str(
            l2_norm_sgpp) + "\n\tLmax: " + str(
            lmax_norm_sgpp) + "\n\t#Points: " + str(numb_points_combi) + "\n\tLambda: " + str(lambd) + "\n")

    file.write("\nCombi 1 " + str(maximum_level) + ":\n")
    file.write(
        "\tResult: " + str(list(result_combi.flatten())) + "\n\tDiff: " + str(list(diff_combi.flatten())) + "\n\tL1: " + str(l1_norm_combi) + "\n\tL2: " + str(l2_norm_combi) + "\n\tLmax: " + str(
            lmax_norm_combi) + "\n\t#Points: " + str(numb_points_combi) + "\n\tLambda: " + str(lambd) + "\n")

    file.write("\nBest component grid 1 " + str(maximum_level) + ":\n")
    file.write("\tLevels and #Points: " + str(components_points)+"\n")
    file.write("\tL1: " + str(l1_norm_components) + "\n\tBest component L1: " + str(best_component_l1) + "\n\tL2: " + str(
        l2_norm_components) + "\n\tBest component L2: " + str(best_component_l2) + "\n\tLmax: " + str(
        lmax_norm_components) + "\n\tBest component Lmax: " + str(best_component_lmax) + "\n\t#Points: " + str(numb_points_combi) + "\n\t#Grids: " + str(
        len(result_components)) + "\n\tLambda: " + str(lambd) + "\n")

    file.write("\nLmin Lmax " + str(minimum_level) + " " + str(maximum_level) + ":\n")
    file.write("\tResult: " + str(list(result_lmin_lmax.flatten())) + "\n\tDiff: " + str(list(diff_lmin_lmax.flatten())) + "\n\tL1: " + str(l1_norm_lmin_lmax) + "\n\tL2: " + str(
        l2_norm_lmin_lmax) + "\n\tLmax: " + str(lmax_norm_lmin_lmax) + "\n\t#Points: " + str(numb_points_lmin_lmax) + "\n\t#Grids: " + str(
        len(combiObject_lmin_lmax.scheme)) + "\n\tLambda: " + str(lambd) + "\n")

    file.write("\nLumping 1 " + str(maximum_level) + ":\n")
    file.write(
        "\tResult: " + str(list(result_lumping.flatten())) + "\n\tDiff: " + str(list(diff_lumping.flatten())) + "\n\tL1: " + str(l1_norm_lumping) + "\n\tL2: " + str(
            l2_norm_lumping) + "\n\tLmax: " + str(lmax_norm_lumping) + "\n\t#Points: " + str(numb_points_lumping) + "\n\tLambda: " + str(lambd) + "\n")

    file.close()
