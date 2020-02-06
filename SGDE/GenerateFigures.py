from sys import path

path.append('../src/')
import numpy as np
from ErrorCalculator import *
from GridOperation import *
from StandardCombi import *
from sklearn import datasets
from SGppCompare import plot_comparison

# dimension of the problem
dim = 2

# define number of samples
size = 500

# define integration domain boundaries
a = np.zeros(dim)
b = np.ones(dim)

datasets = ["Datasets/2D_StroSkewB2.csv", "Datasets/Circles500.csv", "Datasets/faithful.csv", "Datasets/funnychess.csv", "Datasets/twomoons.csv"]
values = {"Datasets/2D_StroSkewB2.csv": ["Values/StroSkewB2_level_3_lambda_0.0.csv", "Values/StroSkewB2_level_3_lambda_0.1.csv",
                                         "Values/StroSkewB2_level_3_lambda_1.0_optimized.csv",
                                         "Values/StroSkewB2_level_4_lambda_0.0.csv", "Values/StroSkewB2_level_4_lambda_0.1.csv",
                                         "Values/StroSkewB2_level_4_lambda_1.0_optimized.csv",
                                         "Values/StroSkewB2_level_5_lambda_0.0.csv", "Values/StroSkewB2_level_5_lambda_0.1.csv",
                                         "Values/StroSkewB2_level_5_lambda_1.0_optimized.csv"],
          "Datasets/Circles500.csv": ["Values/Circles_level_3_lambda_0.0.csv", "Values/Circles_level_3_lambda_0.1.csv",
                                      "Values/Circles_level_3_lambda_1.0_optimized.csv",
                                      "Values/Circles_level_4_lambda_0.0.csv", "Values/Circles_level_4_lambda_0.1.csv",
                                      "Values/Circles_level_4_lambda_1.0_optimized.csv",
                                      "Values/Circles_level_5_lambda_0.0.csv", "Values/Circles_level_5_lambda_0.1.csv",
                                      "Values/Circles_level_5_lambda_1.0_optimized.csv"],
          "Datasets/faithful.csv": ["Values/Faithful_level_3_lambda_0.0.csv", "Values/Faithful_level_3_lambda_0.1.csv",
                                    "Values/Faithful_level_3_lambda_1.0_optimized.csv",
                                    "Values/Faithful_level_4_lambda_0.0.csv", "Values/Faithful_level_4_lambda_0.1.csv",
                                    "Values/Faithful_level_4_lambda_1.0_optimized.csv",
                                    "Values/Faithful_level_5_lambda_0.0.csv", "Values/Faithful_level_5_lambda_0.1.csv",
                                    "Values/Faithful_level_5_lambda_1.0_optimized.csv"],
          "Datasets/funnychess.csv": ["Values/funnyChess_level_3_lambda_0.0.csv", "Values/funnyChess_level_3_lambda_0.1.csv",
                                      "Values/funnyChess_level_3_lambda_1.0_optimized.csv",
                                      "Values/funnyChess_level_4_lambda_0.0.csv", "Values/funnyChess_level_4_lambda_0.1.csv",
                                      "Values/funnyChess_level_4_lambda_1.0_optimized.csv",
                                      "Values/funnyChess_level_5_lambda_0.0.csv", "Values/funnyChess_level_5_lambda_0.1.csv",
                                      "Values/funnyChess_level_5_lambda_1.0_optimized.csv"],
          "Datasets/twomoons.csv": ["Values/Moons_level_3_lambda_0.0.csv", "Values/Moons_level_3_lambda_0.1.csv",
                                    "Values/Moons_level_3_lambda_1.0_optimized.csv",
                                    "Values/Moons_level_4_lambda_0.0.csv", "Values/Moons_level_4_lambda_0.1.csv",
                                    "Values/Moons_level_4_lambda_1.0_optimized.csv",
                                    "Values/Moons_level_5_lambda_0.0.csv", "Values/Moons_level_5_lambda_0.1.csv",
                                    "Values/Moons_level_5_lambda_1.0_optimized.csv"]}
levels = [3, 4, 5]
lambdas = [0.0, 0.1, 1.0]
pairs = [(3, 0.0), (3, 0.1), (3, 1.0), (4, 0.0), (4, 0.1), (4, 1.0), (5, 0.0), (5, 0.1), (5, 1.0)]
for i in range(len(datasets)):
    # csv dataset file
    data = datasets[i]
    valuesSGpp = values.get(data)
    for j in range(len(valuesSGpp)):
        # SGpp values for datasetf
        valuesCurr = valuesSGpp[j]

        # define lambda
        lambd = pairs[j][1]

        # define level of combigrid
        minimum_level = 1
        maximum_level = pairs[j][0]

        # define operation to be performed
        operation = DensityEstimation(data, dim, lambd=lambd)

        # create the combiObject and initialize it with the operation
        combiObject = StandardCombi(a, b, operation=operation)

        # perform the density estimation operation, has to be done before the printing and plotting
        combiObject.perform_operation(minimum_level, maximum_level)

        print("Plot of dataset:")
        operation.plot_dataset("Figures/dataset_" + data[9:-4] + ".png")

        print("Combination Scheme:")
        # when you pass the operation the function also plots the contour plot of each component grid
        combiObject.print_resulting_combi_scheme(
            "Figures/combiScheme_" + data[9:-4] + "_" + str(minimum_level) + "_" + str(maximum_level) + "_" + str(lambd) + ".png",
            operation=operation)

        print("Sparse Grid:")
        combiObject.print_resulting_sparsegrid("Figures/sparseGrid_" + str(minimum_level) + "_" + str(maximum_level) + ".png",
                                               markersize=20)

        print("Plot of density estimation")
        # when contour = True, the contour plot is shown next to the 3D plot
        combiObject.plot("Figures/DEplot_" + data[9:-4] + "_" + str(minimum_level) + "_" + str(maximum_level) + "_" + str(lambd) + ".png",
                         contour=True)

        print("Plot of comparison between sparseSpACE and SG++")
        # plot comparison between sparseSpACE and SG++ result
        plot_comparison(dim=dim, data=data, values=valuesCurr, combiObject=combiObject, plot_data=False, minimum_level=minimum_level,
                        maximum_level=maximum_level, lambd=lambd,
                        pointsPerDim=100)
