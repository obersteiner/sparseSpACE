import numpy as np
from sparseSpACE.Utils import *
from sparseSpACE import Grid
from typing import Tuple, Sequence, Callable
from sparseSpACE.Function import *
from scipy.linalg import solve_triangular

class HierarchizationLSG(object):
    def __init__(self, grid):
        self.grid = grid

    def __call__(self, grid_values: Sequence[Sequence[float]], numPoints: Sequence[int], grid: Grid) -> Sequence[Sequence[float]]:
        self.grid = grid
        self.dim = len(numPoints)
        for d in range(self.dim):
            assert numPoints[d] == len(self.grid.get_coordinates_dim(d))
        for d in range(self.dim):
            grid_values = self.hierarchize_poles_for_dim(grid_values, numPoints, d)
        return grid_values

    # this function applies a one dimensional hierarchization (in dimension d) to the array grid_values with
    # numPoints (array) many points for each dimension
    def hierarchize_poles_for_dim(self, grid_values: Sequence[Sequence[float]], numPoints: Sequence[int], d: int) -> Sequence[Sequence[float]]:
        if numPoints[d] == 1:
            assert math.isclose(self.grid.get_basis(d, 0)(self.grid.get_coordinates_dim(d)[0]), 1.0)
            return grid_values
        self.dim = len(numPoints)
        offsets = np.array([int(np.prod(numPoints[d+1:])) for d in range(self.dim)])
        numPoints_slice = np.array(numPoints)
        numPoints_slice[d] = 1
        value_length = np.shape(grid_values)[0]
        # create all indeces in d-1 dimensional slice
        point_indeces = get_cross_product_range(numPoints_slice)
        # in the first dimension we need to fill it with the actual function values

        # create and fill matrix for linear system of equations
        # evaluate all basis functions at all grid points
        matrix = np.empty((numPoints[d], numPoints[d]))
        for i in range(numPoints[d]):
            for j in range(numPoints[d]):
                matrix[i, j] = self.grid.get_basis(d, j)(self.grid.get_coordinates_dim(d)[i])
        if numPoints[d] >= 15:
            Q, R = np.linalg.qr(matrix)
        #M_inv = np.linalg.inv(matrix)

        pole_coordinates_base = np.empty(numPoints[d], dtype=int)
        for i in range(numPoints[d]):
            pole_index = np.zeros(self.dim, dtype=int)
            pole_index[d] = i
            pole_coordinates_base[i] = self.get_1D_coordinate(pole_index, offsets)

        # iterate over all indeces in slice (0 at dimension d)
        for point_index in point_indeces:
            # create array of function values through pole
            pole_values = np.zeros((value_length, numPoints[d]))
            #print(pole_coordinates_base)
            pole_coordinates = pole_coordinates_base + int(self.get_1D_coordinate(np.asarray(point_index), offsets))
            #print(pole_coordinates)
            # fill pole_values with function or surplus values of pole through current index
            for i in range(numPoints[d]):
                # use previous surplusses for every consecutive dimension (unidirectional principle)
                pole_values[:, i] = grid_values[:, pole_coordinates[i]]

            # solve system of linear equations for all components of our function values (typically scalar -> output.length = 1)
            # if the function outputs vectors then we have to iterate over all components individually
            # toDo replace by LU factorization to save time
            #(matrix, self.grid.get_coordinates_dim(d), d)
            for n in range(value_length):
                #hierarchized_values = np.linalg.solve(matrix, pole_values[n,:])
                #print(hierarchized_values)
                if numPoints[d] >= 15:
                    hierarchized_values = solve_triangular(R, np.inner(Q.T, pole_values[n, :]), check_finite=False)
                else:
                    hierarchized_values = np.linalg.solve(matrix, pole_values[n, :])
                #hierarchized_values = (np.inner(M_inv, pole_values[n,:]))
                #print(hierarchized_values - hierarchized_values2)
                #if (np.sum(hierarchized_values - hierarchized_values2) > 10**-7):
                #    print(hierarchized_values, hierarchized_values2)
                for i in range(numPoints[d]):
                    #pole_index = point_index[:d] + (i,) + point_index[d+1:]
                    grid_values[n,pole_coordinates[i]] = hierarchized_values[i]
        return grid_values

    # this function maps the d-dimensional index to a one-dimensional array index
    def get_1D_coordinate(self, index_vector: Sequence[int], offsets: Sequence[int]) -> int:
        index = np.sum(index_vector*offsets)
        return index
