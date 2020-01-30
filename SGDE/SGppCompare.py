from sys import path

path.append('../src/')
import numpy as np
from ErrorCalculator import *
from GridOperation import *
from StandardCombi import *
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets, preprocessing

# dimension of the problem
dim = 2

# define number of samples
size = 500

# define integration domain boundaries
a = np.zeros(dim)
b = np.ones(dim)

# csv file
data = "Circles500.csv"

# define operation to be performed
operation = DensityEstimation(data, dim)

# create the combiObject and initialize it with the operation
combiObject = StandardCombi(a, b, operation=operation)

# define level of combigrid
minimum_level = 1
maximum_level = 4

# perform the density estimation operation
combiObject.perform_operation(minimum_level, maximum_level)

print("Plot of dataset:")
operation.plot_dataset()

print("Plot of density estimation")
combiObject.plot()

pointsPerDim = 100
X = np.linspace(0.0, 1.0, pointsPerDim)
Y = np.linspace(0.0, 1.0, pointsPerDim)
X, Y = np.meshgrid(X, Y)

print("Plot of Z")
Z = np.zeros_like(X)
for i in range(pointsPerDim):
    for j in range(pointsPerDim):
        Z[i][j] = combiObject([[X[i, j], Y[i, j]]])

fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
ax.set_zlim(bottom=0.0)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()
plt.close(fig)

# read in SGpp values for the above points
dataSGpp = "valuesSGpp.csv"
dataCSV = np.genfromtxt(dataSGpp, delimiter=',')

print("Plot of SGpp")
fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(X, Y, dataCSV, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
ax.set_zlim(bottom=0.0)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()
plt.close(fig)

difValues = np.subtract(Z, dataCSV)

print("Plot of difference between sparseSpACE and SGpp")
fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(X, Y, difValues, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
ax.set_zlim(bottom=0.0)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()
plt.close(fig)
