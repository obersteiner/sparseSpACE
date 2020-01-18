import numpy as np
from numpy.linalg import solve
from scipy.integrate import quad, nquad, simps
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Set, Dict, Tuple, Optional, Union, Sequence, Generator
from itertools import product
from sklearn import cluster, datasets, preprocessing


class gridPoint:

    def __init__(self, index, level):
        self.value = []
        self.index = index
        self.level = level
        self.meshsize = (pow(2, -self.level[0]), pow(2, -self.level[1]))
        self.pos = self.getPosition()
        self.domain = self.getDomain()
        self.func = lambda x: max(1 - abs(x), 0)

    def getPosition(self):
        return ((self.index[0]) * self.meshsize[0], (self.index[1]) * self.meshsize[1])

    def getDomain(self):
        return ((self.pos[0] - self.meshsize[0], self.pos[0] + self.meshsize[0]),
                (self.pos[1] - self.meshsize[1], self.pos[1] + self.meshsize[1]))


class sparseGrid:

    def __init__(self, level=1, dim=1):
        self.level = level
        self.dim = dim
        self.domain = ((0.0, 1.0),) * dim
        self.grid = {}
        self.indices = []


def get_cross_product_list(one_d_arrays: Sequence[Sequence[Union[float, int]]]) -> List[Tuple[Union[float, int], ...]]:
    return list(get_cross_product(one_d_arrays))


def get_cross_product_range(one_d_arrays: Sequence[Sequence[int]]) -> Generator[Tuple[int, ...], None, None]:
    return get_cross_product([range(one_d_array) for one_d_array in one_d_arrays])


def get_cross_product_range_list(one_d_arrays: Sequence[Sequence[int]]) -> List[Tuple[int, ...]]:
    return get_cross_product_list([range(one_d_array) for one_d_array in one_d_arrays])


def get_cross_product(one_d_arrays: Sequence[Sequence[Union[float, int]]]) -> Generator[
    Tuple[Union[float, int], ...], None, None]:
    # dim = len(one_d_arrays)
    # return list(zip(*[g.ravel() for g in
    #          np.meshgrid(*[one_d_arrays[d] for d in range(dim)], indexing="ij")]))
    return product(*one_d_arrays)


def checkIfAdjacent(ivec, jvec):
    for i in range(len(ivec)):
        if abs(ivec[i] - jvec[i]) > 1:
            return False
    return True


# Add g2 to g1
def addGrids(g1, g2):
    for key in g2:
        coord, pos = key
        if pos not in g1:
            g1.update({key: g2.get(key)})
    return g1


# Subtract g2 from g1
def subGrids(g1, g2):
    for key in g2:
        coord, pos = key
        if pos in g1:
            g1.pop(pos)
    return g1


def merge(grid):
    ret = {}
    for row in grid:
        for i in range(0, len(row)):
            ret.update({(row[i].index, row[i].pos): row[i]})
    return ret


def getPosition(index, levelvec):
    meshsize = (pow(2, -levelvec[0]), pow(2, -levelvec[1]))
    return ((index[1]) * meshsize[0], (index[0]) * meshsize[1])


def calcDomain(pos, levelvec):
    meshsize = (pow(2, -levelvec[0]), pow(2, -levelvec[1]))
    return [(pos[0] - meshsize[0], pos[1]), (pos[0] + meshsize[0], pos[1]), (pos[0], pos[1] - meshsize[1]),
            (pos[0], pos[1] + meshsize[1])]


def getSharedDomain(ivec, jvec, levelvec):
    meshsize = (pow(2, -levelvec[0]), pow(2, -levelvec[1]))
    ipos = getPosition(ivec, levelvec)
    jpos = getPosition(jvec, levelvec)
    return [[(min(ipos[0], jpos[0])) - meshsize[0], (min(ipos[0], jpos[0])) + meshsize[0]],
            [(max(ipos[1], jpos[1])) - meshsize[1], (min(ipos[0], jpos[0])) + meshsize[1]]]


def getCoordPosMapping(grid: dict):
    keys = list(grid.keys())
    ret = {}
    for i in range(len(keys)):
        ret.update({keys[i][0]: keys[i][1]})
    return ret


def getIndexList(levelvec):
    lengths = [2 ** l - 1 for l in levelvec]
    dim_lists = [range(1, n + 1) for n in lengths]
    return get_cross_product_list(dim_lists)


def calcRange(ivec, jvec, levelvec):
    return [(2 ** (-levelvec[0])) * (min(ivec[1], jvec[1]) - 1), 2 ** (-levelvec[0]) * (max(ivec[0], jvec[0]) + 1)]


def calcNumbOfPoints(levelvec):
    ret = 1
    for i in range(len(levelvec)):
        ret *= 2 ** levelvec[i] - 1
    return ret


def constructGrid(levelvec):
    grid = np.ndarray((pow(2, levelvec[1]) - 1, (pow(2, levelvec[0]) - 1)), dtype=gridPoint)
    shape = grid.shape
    indexes = getIndexList(levelvec)
    i = 1
    for row in grid:
        for x in range(0, shape[1]):
            row[x] = gridPoint(indexes.pop(0), levelvec)
            i = i + 1
            # entry = hat(pow(2, levelvec[0]))
    return merge(grid)


def construct1D(level):
    grid = np.arange(1, pow(2, level))
    subspace = []
    for x in grid:
        # subspace.append((x * pow(2, -level), hat(0.5,level)))
        subspace.append((x * pow(2, -level), hat(0)))
    return subspace


def construct1DHier(level):
    temp = np.arange(1, pow(2, level))
    grid = temp[temp % 2 == 1]
    subspace = []
    for x in grid:
        # subspace.append((x * pow(2, -level), hat(0.5,level)))
        subspace.append((x * pow(2, -level), hat(0)))
    return subspace


def combiTech(level):
    gridsAdd = []
    gridsSub = []
    for i in range(level, 0, -1):
        for j in range(1, level + 1, 1):
            if (i + j == level + 1):
                gridsAdd.append(constructGrid((i, j)))

    for k in range(level - 1, 0, -1):
        for l in range(1, level, 1):
            if (k + l == level):
                gridsSub.append(constructGrid((k, l)))

    ret = gridsAdd[0]
    for m in range(len(gridsAdd) - 1):
        ret = addGrids(ret, gridsAdd[m + 1])

    return ret


def getL2ScalarProduct(ivec, jvec, lvec):
    if checkIfAdjacent(ivec, jvec):
        dim = len(ivec)
        f = lambda x, y: (hat3(ivec, lvec, [x, y]) * hat3(jvec, lvec, [x, y]))
        print("-------------")
        print("calc")
        start = [(min(ivec[d], jvec[d]) - 1) * 2 ** (-lvec[d]) for d in range(dim)]
        end = [(max(ivec[d], jvec[d]) + 1) * 2 ** (-lvec[d]) for d in range(dim)]
        print(ivec, jvec)
        print(start, end)
        return nquad(f, [[start[d], end[d]] for d in range(dim)], opts={"epsabs": 10 ** (-15),
                                                                        "epsrel": 1 ** (
                                                                            -15)})  # nquad(f, getSharedDomain(ivec, jvec, lvec))
    else:
        print("-------------")
        print("skipped")
        return (0, 0)


def calcDensityEstimation(levelvec, data, lambd=0):
    grid = constructGrid(levelvec)
    R = constructR(levelvec, grid)
    R[np.diag_indices_from(R)] += lambd
    b = calcB(data, levelvec)
    return solve(R, b)


def calcB(data, levelvec):
    M = len(data)
    N = calcNumbOfPoints(levelvec)
    b = np.empty(N)

    indexList = getIndexList(levelvec)
    for i in range(N):
        sum = 0
        for j in range(M):
            sum += hat3(indexList[i], levelvec, data[j])
        b[i] = ((1 / M) * sum)
        print((1 / M) * sum)
    return b


def constructR(levelvec, grid, lumping=False):
    coords = getCoordPosMapping(grid)
    numberPoints = len(coords)
    R = np.zeros((numberPoints, numberPoints))
    indexList = getIndexList(levelvec)
    print(indexList, levelvec)
    diagVal, err = getL2ScalarProduct(indexList[0], indexList[0], levelvec)
    R[np.diag_indices_from(R)] += diagVal
    if lumping == False:
        for i in range(0, len(indexList) - 1):
            for j in range(i + 1, len(indexList)):
                temp, err = getL2ScalarProduct(indexList[i], indexList[j], levelvec)
                print(i, j)
                print(temp)

                if temp != 0:
                    R[i, j] = temp
                    R[j, i] = temp
    return R


def calcAlphas(level, data, lambd=0):
    alphas = {}
    for i in range(level, 0, -1):
        for j in range(1, level + 1, 1):
            if i + j == level + 1:
                print("Calculating grid add", (i, j), sep=" ")
                alphas.update({(i, j): calcDensityEstimation((i, j), data, lambd)})
    for k in range(level - 1, 0, -1):
        for l in range(1, level, 1):
            if k + l == level:
                print("Calculating grid sub", (k, l), sep=" ")
                alphas.update({(k, l): calcDensityEstimation((k, l), data, lambd)})
    return alphas


def preCalcAlphas(upToLevel, data, lambd=0):
    alphas = {}
    # alphas.update({1: {(1, 1): calcDensityEstimation((1, 1), data, lambd)}})
    for i in range(2, upToLevel + 1):
        alphas.update({i: calcAlphas(i, data, lambd)})
    return alphas


def combiDensityEstimation(level, alphas, x):
    ret = 0
    for i in range(level, 0, -1):
        for j in range(1, level + 1, 1):
            if i + j == level + 1:
                ret += hat4((i, j), alphas.get((i, j)), x)
    for k in range(level - 1, 0, -1):
        for l in range(1, level, 1):
            if k + l == level:
                ret -= hat4((k, l), alphas.get((k, l)), x)
    return ret


def hat(x):
    return np.maximum(1 - abs(x), 0)


def hat2(x, y):
    return np.maximum(1 - abs(x), 0) * np.maximum(1 - abs(y), 0)


def hat3(ivec, lvec, x):
    dim = len(ivec)
    result = 1
    for d in range(dim):
        # TODO Which one?
        # result *= max((1 - abs(2 ** lvec[d] * x - ivec[d])) / (2 ** lvec[d]), 0)
        result *= max((1 - abs(2 ** lvec[d] * x[d] - ivec[d])), 0)
    return result


def hat4(levelvec, alphas, x):
    indices = getIndexList(levelvec)
    sum = 0
    for i, index in enumerate(indices):
        # TODO test if i is in support of x, if not --> dont calculate
        sum += hat3(index, levelvec, x) * alphas[i]
    return sum


# PLOTTING
def plot1D(grid):
    x, y = zip(*grid)
    x = list(x)
    y = list(y)

    ycord = np.tile([0, 1, 0], len(x))
    xcord = np.zeros(ycord.shape)
    c = 0
    for i in range(0, len(xcord), 3):
        xcord[i] = c * x[0]
        xcord[i + 1] = (c + 1) * x[0]
        xcord[i + 2] = (c + 2) * x[0]
        c = c + 1
        plt.plot(xcord[i:(i + 3)], ycord[i:(i + 3)])

    plt.axis([0, 1, 0, 2])
    plt.show()


def plot1DHier(grid):
    x, y = zip(*grid)
    x = list(x)
    y = list(y)

    ycord = np.tile([0, 1], len(x))
    ycord = np.append(ycord, 0)
    xcord = np.zeros(ycord.shape)

    for i in range(len(xcord) - 1):
        xcord[(i + 1)] = (i + 1) * x[0]

    plt.plot(xcord, ycord)
    plt.axis([0, 1, 0, 2])
    plt.show()


def plotGrid(grid):
    points = grid.keys()
    indices = list(map(lambda x: x[0], points))
    cords = list(map(lambda x: x[1], points))
    x, y = zip(*cords)

    plt.scatter(x, y, s=100, color="black")
    for i, txt in enumerate(indices):
        plt.annotate(txt, (x[i] + 0.01, y[i] + 0.01))

    xticks = [0, 1]
    xticks[1:1] = x
    yticks = [0, 1]
    yticks[1:1] = y
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.axis([0, 1, 0, 1])
    # plt.title("Grid level")
    plt.grid(True)
    plt.show()


def plotGridWithDomain(grid, levelvec):
    points = grid.keys()
    indices = list(map(lambda x: x[0], points))
    cords = list(map(lambda x: x[1], points))
    x, y = zip(*cords)
    z = []
    plt.scatter(x, y, s=125, color="black")
    for i, txt in enumerate(indices):
        plt.annotate(txt, (x[i] + 0.01, y[i] + 0.01))
        z.extend(calcDomain(cords[i], levelvec))

    a, b = zip(*z)
    plt.scatter(a, b, s=50, color="red", marker="D")
    xticks = [0, 1]
    xticks[1:1] = x
    yticks = [0, 1]
    yticks[1:1] = y
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.axis([0, 1, 0, 1])
    # plt.title("Grid level")
    plt.grid(True)
    plt.show()


def plotCombi(grid):
    points = grid.values()
    cords = list(map(lambda x: tuple(x.pos), points))
    x, y = zip(*cords)

    plt.scatter(x, y, s=100, color="black")
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.axis([0, 1, 0, 1])
    plt.show()


def plotHat3D(index, levelvec, pointsPerDim=100):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X = np.linspace(0.0, 1.0, pointsPerDim)
    Y = np.linspace(0.0, 1.0, pointsPerDim)
    X, Y = np.meshgrid(X, Y)

    Z = np.zeros_like(X)
    for i in range(pointsPerDim):
        for j in range(pointsPerDim):
            Z[i][j] = hat3(index, levelvec, [X[i, j], Y[i, j]])

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlim(0.0, 1.01)
    ax.set_ylim(0.0, 1.01)
    ax.set_zlim(0.0, 1.01)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()
    plt.close(fig)


def plotGrid3D(levelvec, pointsPerDim=100):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    shape = (pow(2, levelvec[1]) - 1, (pow(2, levelvec[0]) - 1))
    indexList = getIndexList(levelvec)
    drawList = []
    for v in range(shape[0]):
        temp = indexList[v * shape[0]:v * shape[0] + shape[0]]
        temp.reverse()
        drawList.extend(temp)
    X = np.linspace(0.0, 1.0, pointsPerDim)
    Y = np.linspace(0.0, 1.0, pointsPerDim)
    X, Y = np.meshgrid(X, Y)

    # Z = np.zeros_like(X)
    #
    # for k in range(shape[0] * shape[1]):
    #     for i in range(pointsPerDim):
    #         for j in range(pointsPerDim):
    #             Z[i][j] = hat3(drawList[k], levelvec, [X[i, j], Y[i, j]])
    #     surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    Z = np.zeros_like(X)
    temp = np.zeros_like(X)
    for k in range(shape[0] * shape[1]):
        for i in range(pointsPerDim):
            for j in range(pointsPerDim):
                Z[i][j] = hat3(indexList[k], levelvec, [X[i, j], Y[i, j]])
        temp = np.add(temp, Z)
    surf = ax.plot_surface(X, Y, temp, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlim(0.0, 1.01)
    ax.set_ylim(0.0, 1.01)
    ax.set_zlim(0.0, 1.01)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()


def plot(level, data, pointsPerDim=100):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data[0])
    dataTrans = scaler.transform(data[0])
    alphas = calcAlphas(level, dataTrans, 0)

    X = np.linspace(0.0, 1.0, pointsPerDim)
    Y = np.linspace(0.0, 1.0, pointsPerDim)
    X, Y = np.meshgrid(X, Y)

    Z = np.zeros_like(X)
    for i in range(pointsPerDim):
        for j in range(pointsPerDim):
            Z[i][j] = combiDensityEstimation(level, alphas, [X[i, j], Y[i, j]])

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


def plotData(data):
    points = data[0]
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaler.fit(points)
    dataTrans = scaler.transform(points)
    x, y = zip(*dataTrans)
    plt.scatter(x, y)
    plt.title("#points = %d" % len(points))
    plt.show()


moons = datasets.make_moons(noise=0.05)
circle = datasets.make_circles(noise=0.05)
# alphas = preCalcAlphas(3, moons, 0)

oldFaithfulDataset = ([[3.600, 79],
                       [1.800, 54],
                       [3.333, 74],
                       [2.283, 62],
                       [4.533, 85],
                       [2.883, 55],
                       [4.700, 88],
                       [3.600, 85],
                       [1.950, 51],
                       [4.350, 85],
                       [1.833, 54],
                       [3.917, 84],
                       [4.200, 78],
                       [1.750, 47],
                       [4.700, 83],
                       [2.167, 52],
                       [1.750, 62],
                       [4.800, 84],
                       [1.600, 52],
                       [4.250, 79],
                       [1.800, 51],
                       [1.750, 47],
                       [3.450, 78],
                       [3.067, 69],
                       [4.533, 74],
                       [3.600, 83],
                       [1.967, 55],
                       [4.083, 76],
                       [3.850, 78],
                       [4.433, 79],
                       [4.300, 73],
                       [4.467, 77],
                       [3.367, 66],
                       [4.033, 80],
                       [3.833, 74],
                       [2.017, 52],
                       [1.867, 48],
                       [4.833, 80],
                       [1.833, 59],
                       [4.783, 90],
                       [4.350, 80],
                       [1.883, 58],
                       [4.567, 84],
                       [1.750, 58],
                       [4.533, 73],
                       [3.317, 83],
                       [3.833, 64],
                       [2.100, 53],
                       [4.633, 82],
                       [2.000, 59],
                       [4.800, 75],
                       [4.716, 90],
                       [1.833, 54],
                       [4.833, 80],
                       [1.733, 54],
                       [4.883, 83],
                       [3.717, 71],
                       [1.667, 64],
                       [4.567, 77],
                       [4.317, 81],
                       [2.233, 59],
                       [4.500, 84],
                       [1.750, 48],
                       [4.800, 82],
                       [1.817, 60],
                       [4.400, 92],
                       [4.167, 78],
                       [4.700, 78],
                       [2.067, 65],
                       [4.700, 73],
                       [4.033, 82],
                       [1.967, 56],
                       [4.500, 79],
                       [4.000, 71],
                       [1.983, 62],
                       [5.067, 76],
                       [2.017, 60],
                       [4.567, 78],
                       [3.883, 76],
                       [3.600, 83],
                       [4.133, 75],
                       [4.333, 82],
                       [4.100, 70],
                       [2.633, 65],
                       [4.067, 73],
                       [4.933, 88],
                       [3.950, 76],
                       [4.517, 80],
                       [2.167, 48],
                       [4.000, 86],
                       [2.200, 60],
                       [4.333, 90],
                       [1.867, 50],
                       [4.817, 78],
                       [1.833, 63],
                       [4.300, 72],
                       [4.667, 84],
                       [3.750, 75],
                       [1.867, 51],
                       [4.900, 82],
                       [2.483, 62],
                       [4.367, 88],
                       [2.100, 49],
                       [4.500, 83],
                       [4.050, 81],
                       [1.867, 47],
                       [4.700, 84],
                       [1.783, 52],
                       [4.850, 86],
                       [3.683, 81],
                       [4.733, 75],
                       [2.300, 59],
                       [4.900, 89],
                       [4.417, 79],
                       [1.700, 59],
                       [4.633, 81],
                       [2.317, 50],
                       [4.600, 85],
                       [1.817, 59],
                       [4.417, 87],
                       [2.617, 53],
                       [4.067, 69],
                       [4.250, 77],
                       [1.967, 56],
                       [4.600, 88],
                       [3.767, 81],
                       [1.917, 45],
                       [4.500, 82],
                       [2.267, 55],
                       [4.650, 90],
                       [1.867, 45],
                       [4.167, 83],
                       [2.800, 56],
                       [4.333, 89],
                       [1.833, 46],
                       [4.383, 82],
                       [1.883, 51],
                       [4.933, 86],
                       [2.033, 53],
                       [3.733, 79],
                       [4.233, 81],
                       [2.233, 60],
                       [4.533, 82],
                       [4.817, 77],
                       [4.333, 76],
                       [1.983, 59],
                       [4.633, 80],
                       [2.017, 49],
                       [5.100, 96],
                       [1.800, 53],
                       [5.033, 77],
                       [4.000, 77],
                       [2.400, 65],
                       [4.600, 81],
                       [3.567, 71],
                       [4.000, 70],
                       [4.500, 81],
                       [4.083, 93],
                       [1.800, 53],
                       [3.967, 89],
                       [2.200, 45],
                       [4.150, 86],
                       [2.000, 58],
                       [3.833, 78],
                       [3.500, 66],
                       [4.583, 76],
                       [2.367, 63],
                       [5.000, 88],
                       [1.933, 52],
                       [4.617, 93],
                       [1.917, 49],
                       [2.083, 57],
                       [4.583, 77],
                       [3.333, 68],
                       [4.167, 81],
                       [4.333, 81],
                       [4.500, 73],
                       [2.417, 50],
                       [4.000, 85],
                       [4.167, 74],
                       [1.883, 55],
                       [4.583, 77],
                       [4.250, 83],
                       [3.767, 83],
                       [2.033, 51],
                       [4.433, 78],
                       [4.083, 84],
                       [1.833, 46],
                       [4.417, 83],
                       [2.183, 55],
                       [4.800, 81],
                       [1.833, 57],
                       [4.800, 76],
                       [4.100, 84],
                       [3.966, 77],
                       [4.233, 81],
                       [3.500, 87],
                       [4.366, 77],
                       [2.250, 51],
                       [4.667, 78],
                       [2.100, 60],
                       [4.350, 82],
                       [4.133, 91],
                       [1.867, 53],
                       [4.600, 78],
                       [1.783, 46],
                       [4.367, 77],
                       [3.850, 84],
                       [1.933, 49],
                       [4.500, 83],
                       [2.383, 71],
                       [4.700, 80],
                       [1.867, 49],
                       [3.833, 75],
                       [3.417, 64],
                       [4.233, 76],
                       [2.400, 53],
                       [4.800, 94],
                       [2.000, 55],
                       [4.150, 76],
                       [1.867, 50],
                       [4.267, 82],
                       [1.750, 54],
                       [4.483, 75],
                       [4.000, 78],
                       [4.117, 79],
                       [4.083, 78],
                       [4.267, 78],
                       [3.917, 70],
                       [4.550, 79],
                       [4.083, 70],
                       [2.417, 54],
                       [4.183, 86],
                       [2.217, 50],
                       [4.450, 90],
                       [1.883, 54],
                       [1.850, 54],
                       [4.283, 77],
                       [3.950, 79],
                       [2.333, 64],
                       [4.150, 75],
                       [2.350, 47],
                       [4.933, 86],
                       [2.900, 63],
                       [4.583, 85],
                       [3.833, 82],
                       [2.083, 57],
                       [4.367, 82],
                       [2.133, 67],
                       [4.350, 74],
                       [2.200, 54],
                       [4.450, 83],
                       [3.567, 73],
                       [4.500, 73],
                       [4.150, 88],
                       [3.817, 80],
                       [3.917, 71],
                       [4.450, 83],
                       [2.000, 56],
                       [4.283, 79],
                       [4.767, 78],
                       [4.533, 84],
                       [1.850, 58],
                       [4.250, 83],
                       [1.983, 43],
                       [2.250, 60],
                       [4.750, 75],
                       [4.117, 81],
                       [2.150, 46],
                       [4.417, 90],
                       [1.817, 46],
                       [4.467, 74]], 0)

plotData(oldFaithfulDataset)
plotData(moons)
plotData(circle)
plot(4, oldFaithfulDataset)
plot(4, moons)
plot(4, circle)

# alphas = calcAlphas(3, circle, 0)
# alpha = calcDensityEstimation((2, 2), circle, 0)

# plotGrid3D((2, 2))
# plotHat3D((2, 2), (2, 2))

# R = constructR((2, 2), constructGrid((2, 2)))

# plotGridWithDomain(constructGrid((2, 2)), (2, 2))
# plot1DHier(construct1DHier(1))
# plot1DHier(construct1DHier(2))
# plot1DHier(construct1DHier(3))
# plot1D(construct1D(1))
# plot1D(construct1D(2))
# plot1D(construct1D(3))
#
# plotGrid(constructGrid((3, 1)))
# plotGrid(constructGrid((2, 2)))
# plotGrid(constructGrid((1, 3)))
# plotGrid(constructGrid((1, 2)))
# plotGrid(constructGrid((2, 1)))
#
# plotCombi(combiTech(3))
# plotCombi(combiTech(4))
# plotCombi(combiTech(5))
