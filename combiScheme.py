import numpy as np
import math


def getCombiScheme(lmin, lmax, dim, do_print=True):
    grid_array = []
    for q in range(min(dim, lmax-lmin+1)):
        coefficient = (-1)**q * math.factorial(dim-1)/(math.factorial(q)*math.factorial(dim-1-q))
        grids = getGrids(dim, lmax-q)
        grid_array.extend([(np.array(g, dtype=int)+np.ones(dim, dtype=int)*(lmin-1), coefficient) for g in grids])
    for i in range(len(grid_array)):
        if do_print:
            print(i, list(grid_array[i][0]), grid_array[i][1])
    return grid_array


def getGrids(dim_left, values_left):
    if dim_left == 1:
        return [[values_left]]
    grids = []
    for index in range(values_left):
        levelvector = [index+1]
        grids.extend([levelvector + g for g in getGrids(dim_left - 1, values_left - index)])
    return grids
