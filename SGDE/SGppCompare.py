from sys import path

path.append('../src/')
import numpy as np
from ErrorCalculator import *
from GridOperation import *
from StandardCombi import *
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets, preprocessing
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_comparison(filename: str = None, dim: int = 2, data: str = None, values: str = None, combiObject: StandardCombi = None, plot_data: bool = False, minimum_level: int = 1,
                    maximum_level: int = 5, lambd: float = 0.0, pointsPerDim: int = 100):
    if values is None:
        print("No values for comparison given.")
        return
    if combiObject is None and data is not None:
        # define integration domain boundaries
        a = np.zeros(dim)
        b = np.ones(dim)

        # define operation to be performed
        operation = DensityEstimation(data, dim, lambd=lambd)

        # create the combiObject and initialize it with the operation
        combi = StandardCombi(a, b, operation=operation)

        # perform the density estimation operation
        combi.perform_operation(minimum_level, maximum_level)
    elif combiObject is not None:
        combi = combiObject
    else:
        print("No data or combiObject given.")
        return

    if plot_data:
        print("Plot of dataset:")
        operation.plot_dataset("Figures/dataset_" + data[9:-4] + "_" + str(minimum_level) + "_" + str(maximum_level) + "_" + str(lambd) + ".png")

    # print("Plot of density estimation")
    # combiObject.plot(contour=True)

    X = np.linspace(0.0, 1.0, pointsPerDim)
    Y = np.linspace(0.0, 1.0, pointsPerDim)
    X, Y = np.meshgrid(X, Y)

    Z = combi(list(map(lambda x, y: (x, y), X.flatten(), Y.flatten())))
    Z = Z.reshape((100, 100))

    fontsize = 30
    plt.rcParams.update({'font.size': fontsize})

    fig = plt.figure(figsize=(30, 20))
    ax = fig.add_subplot(2, 3, 1, projection='3d')
    ax.title.set_text("sparseSpACE")
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax = fig.add_subplot(2, 3, 4)
    p = ax.imshow(Z, extent=[0.0, 1.0, 0.0, 1.0], origin='lower', cmap=cm.coolwarm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(p, cax=cax)

    # read in SGpp values for the above points
    dataCSV = np.genfromtxt(values, delimiter=',')

    ax = fig.add_subplot(2, 3, 2, projection='3d')
    ax.title.set_text("SG++")
    ax.plot_surface(X, Y, dataCSV, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax = fig.add_subplot(2, 3, 5)
    p = ax.imshow(dataCSV, extent=[0.0, 1.0, 0.0, 1.0], origin='lower', cmap=cm.coolwarm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(p, cax=cax)

    difValues = np.subtract(Z, dataCSV)

    ax = fig.add_subplot(2, 3, 3, projection='3d')
    ax.title.set_text("Difference")
    ax.plot_surface(X, Y, difValues, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax = fig.add_subplot(2, 3, 6)
    p = ax.imshow(difValues, extent=[0.0, 1.0, 0.0, 1.0], origin='lower', cmap=cm.coolwarm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(p, cax=cax)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.rcParams.update({'font.size': plt.rcParamsDefault.get('font.size')})
