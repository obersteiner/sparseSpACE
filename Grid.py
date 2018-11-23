import numpy as np
import abc,logging
from Integrator import *

# the grid class provides basic functionalities for an abstract grid
class Grid(object):

    def __init__(self, a, b, boundary=True):
        self.boundary = boundary
        self.a = a
        self.b = b

    # integrates the grid on the specified area for function f
    def integrate(self, f, levelvec, start, end):
        if not self.isGlobal():
            self.setCurrentArea(start, end, levelvec)
        return self.integrator(f, self.levelToNumPoints(levelvec), start, end)

    #def integrate_point(self, f, levelvec, start, end, point):
    #    if not self.isGlobal():
    #        self.setCurrentArea(start, end, levelvec)
    #    return self.integrator.integrate_point(f, point)

    # the default case is that a grid is nested; overwrite this if not nested!
    def isNested(self):
        return True

    # the default case is that a grid is not globally but only locally defined
    def isGlobal(self):
        return False

    # this method can be used to generate a local grid for the given area and the specified number of points
    # typically the local grid is constructed for each area in the refinement graph during the algorithm
    @abc.abstractmethod
    def setCurrentArea(self, start, end, levelvec):
        pass

    # this method returns the actual coordinate of a point specified by the indexvector of the point
    # (e.g. in a 3 by 3 equdistant grid on the unit interval, the point [1,1] would have coordinate (0.5,0.5) and point [0,2] coordinate (0,1)
    # overwrite if necessary
    def getCoordinate(self, indexvector):
        position = np.empty(self.dim)
        for d in range(self.dim):
            position[d] = self.coordinate_array[d][indexvector[d]]
        return position

    # this method returns all the coordinate tuples of all points in the grid
    @abc.abstractmethod
    def getPoints(self):
        pass

    # this method returns the quadrature weight for the point specified by the indexvector
    @abc.abstractmethod
    def getWeight(self, indexvector):
        pass

    # this method returns the number of points in the grid that correspond to the specified levelvector
    @abc.abstractmethod
    def levelToNumPoints(self, levelvector):
        pass

    # this method translates a point in an equidistant mesh of level self.levelvec to its corresponding index
    def getIndexTo1DCoordinate(self, coordinate, level):
        return coordinate * 2 ** level

    def point_not_zero(self, p):
        #print(p, self.grid.boundary or not (self.point_on_boundary(p)))
        return self.boundary or not (self.point_on_boundary(p))

    def point_on_boundary(self, p):
        #print("2",p, (p == self.a).any() or (p == self.b).any())
        return (p == self.a).any() or (p == self.b).any()

    def get_points_and_weights(self):
        return self.getPoints(), self.get_weights()

    @abc.abstractmethod
    def get_weights(self):
        return list(self.getWeight(index) for index in zip(*[g.ravel() for g in np.meshgrid(*[range(self.numPoints[d]) for d in range(self.dim)])]))


from scipy.optimize import fmin
from scipy.special import eval_hermitenorm, eval_sh_legendre


# this class generates a Leja grid which constructs 1D Leja grid structures
# and constructs the tensorized grid according to the levelvector
class LejaGrid(Grid):
    def __init__(self, boundary=True):
        self.boundary = boundary
        self.integrator = IntegratorArbitraryGrid(self)
        self.linear_growth_factor = 2

    def setCurrentArea(self, start, end, levelvec):
        self.start = start
        self.end = end
        self.dim = len(start)
        self.levelvec = levelvec
        self.numPoints = self.levelToNumPoints(levelvec)
        self.coordinate_array = []
        self.weights = []
        self.length = np.array(end) - np.array(start)
        # prepare coordinates and weights
        for d in range(self.dim):
            coordsD = self.get_1D_level_points(levelvec[d], 0, 1)
            # print(coordsD)
            weightsD = np.array(self.compute_1D_quad_weights(coordsD)) * self.length[d]
            coordsD = np.array(coordsD)
            coordsD *= self.length[d]
            coordsD += self.start[d]
            self.coordinate_array.append(coordsD)
            self.weights.append(weightsD)
        # print(coords)

    def levelToNumPoints(self, levelvec):
        numPoints = np.zeros(len(levelvec), dtype=int)
        for d in range(len(levelvec)):
            numPoints[d] = self.levelToNumPoints1D(levelvec[d])
        return numPoints

    def levelToNumPoints1D(self, level):
        if level == 0:
            numPoints = 1
        else:
            numPoints = self.linear_growth_factor * (level + 1) - 1
        return numPoints

    def getPoints(self):
        return list(zip(*[g.ravel() for g in np.meshgrid(*self.coordinate_array)]))

    def getWeight(self, indexvector):
        weight = 1
        for d in range(self.dim):
            weight *= self.weights[d][indexvector[d]]
        return weight

    def compute_1D_quad_weights(self, grid_1D):
        N = len(grid_1D)
        V = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                V[i, j] = eval_sh_legendre(j, grid_1D[i]) * np.sqrt(2 * j + 1)

        weights = np.linalg.inv(V)[0, :]

        return weights

    def __minimize_function(self, func, a, b):

        guess = a + (b - a) / 2.0
        f_min = fmin(func, guess, xtol=1e-14, maxiter=10000, disp=False)[-1]

        return f_min

    def __get_lleja_poly(self, x, sorted_points, a, b, weightFunction=lambda x: 1.0 / s):

        if (x < a or x > b):
            return -1

        poly = 1.0
        for i in range(len(sorted_points)):
            poly *= np.abs(x - sorted_points[i])

        poly *= weightFunction(x)

        return poly

    def __get_neg_lleja_poly(self, x, sorted_points, a, b, weightFunction=lambda x: 1.0):

        return (-1.0) * self.__get_lleja_poly(x, sorted_points, a, b, weightFunction)

    def __get_starting_point(self, a, b, weightFunction=lambda x: 1.0):

        neg_weight = lambda x: -weightFunction(x)
        starting_point = self.__minimize_function(neg_weight, a, b)

        return starting_point

    def get_1D_level_points(self, curr_level, left_bound, right_bound, weightFunction=lambda x: 1.0, eps=1e-14):

        sorted_points = []
        unsorted_points = []
        no_points = self.levelToNumPoints1D(curr_level)

        starting_point = self.__get_starting_point(left_bound, right_bound, weightFunction)
        sorted_points.append(starting_point)
        unsorted_points.append(starting_point)

        for point in range(1, no_points):
            x_val = []
            y_val = []

            a = 0.
            b = left_bound
            for i in range(len(sorted_points) + 1):
                sorted_points = sorted(sorted_points)

                a = b
                if i < len(sorted_points):
                    b = sorted_points[i]
                else:
                    b = right_bound

                x_min = (a + b) / 2.0
                y_min = 0.0

                if np.abs(b - a) > eps:
                    wlleja_func = lambda x: self.__get_neg_lleja_poly(x, sorted_points, a, b, weightFunction)
                    x_min = self.__minimize_function(wlleja_func, a, b)

                    x_val.append(x_min)
                    y_min = wlleja_func(x_val[-1])
                    y_val.append(y_min)
                else:
                    x_val.append(x_min)
                    y_val.append(y_min)

            wlleja_point = x_val[y_val.index(np.min(y_val))]
            sorted_points.append(wlleja_point)
            unsorted_points.append(wlleja_point)

        unsorted_points = np.array(unsorted_points, dtype=np.float64)

        return unsorted_points




# this class provides an equdistant mesh and uses the trapezoidal rule compute the quadrature
class TrapezoidalGrid(Grid):
    def __init__(self, a, b, boundary=True):
        self.a = a
        self.b = b
        self.boundary = boundary
        self.integrator = IntegratorArbitraryGrid(self)

    def levelToNumPoints(self, levelvec):
        return [2 ** levelvec[d] + 1 - (1 if self.boundary == False else 0) * (int(1 if self.start[d] == self.a[0] else 0) + int(1 if self.end[d] == self.b[0] else 0))
                for d in range(self.dim)]

    def setCurrentArea(self, start, end, levelvec):
        # start of interval
        self.start = start
        # end of interval
        self.end = end
        # level vector per dimension
        self.levelvec = levelvec
        self.dim = len(levelvec)
        # number of points per dimensin
        self.numPoints = self.levelToNumPoints(levelvec)
        self.numPointsWithBoundary = [2 ** levelvec[d] + 1 for d in range(self.dim)]
        # lower and upper border define the subregion of points in the grid with boundary points (only relevant if boundary=False specified)
        # for a grid without boundaries that has one edge at the global boundary we for example reduce the number of points by 1
        # lower border indicates if boundary is at lower end; upper border if boundary is at upper end
        self.lowerBorder = np.zeros(self.dim, dtype=int)
        self.upperBorder = np.array(self.numPoints, dtype=int)
        if not self.boundary:
            for i in range(self.dim):
                if start[i] == self.a[i]:
                    self.lowerBorder[i] = 1
                if end[i] == self.b[i]:
                    self.upperBorder[i] = self.numPointsWithBoundary[i] - 1
                else:
                    self.upperBorder[i] = self.numPointsWithBoundary[i]

        # spacing between two points in each dimension
        self.spacing = (np.array(end) - np.array(start)) / (np.array(self.numPointsWithBoundary) - np.ones(self.dim))
        self.coordinate_array = []
        for d in range(self.dim):
            coordinates = np.empty(self.numPoints[d])
            for i in range(self.numPoints[d]):
                coordinates[i] = self.start[d] + (i + self.lowerBorder[d]) * self.spacing[d]
            self.coordinate_array.append(coordinates)
        self.weight_base = np.prod(self.spacing)

    # return equidistant points generated with numpy.linspace
    def getPoints(self):
        # print(self.numPointsWithBoundary,self.numPoints,list(zip(*[g.ravel() for g in np.meshgrid(*[np.linspace(self.start[i],self.end[i],self.numPointsWithBoundary[i])[self.lowerBorder[i]:self.upperBorder[i]] for i in range(self.dim)])])))
        return list(zip(*[g.ravel() for g in np.meshgrid(*[
            np.linspace(self.start[i], self.end[i], self.numPointsWithBoundary[i])[
            self.lowerBorder[i]:self.upperBorder[i]] for i in range(self.dim)])]))

    '''
    def getCoordinate(self, indexvector):
        position = np.zeros(self.dim)
        for d in range(self.dim):
            position[d] = self.start[d] + (indexvector[d] + self.lowerBorder[d]) * self.spacing[d]
        return position
        '''

    def getWeight(self, indexvector):

        factor = 0  # if point is at the border volume is halfed for each of border dimension
        for d in range(self.dim):
            if indexvector[d] + self.lowerBorder[d] == 0 or indexvector[d] + self.lowerBorder[d] == \
                    self.numPointsWithBoundary[d] - 1:
                factor += 1
        return self.weight_base * 2 ** -factor


# this class generates a grid according to the roots of Chebyshev points and applies a Clenshaw Curtis quadrature
# the formulas are taken from: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.33.3141&rep=rep1&type=pdf
class ClenshawCurtisGrid(Grid):
    def __init__(self, a, b, boundary=True):
        self.a = a
        self.b = b
        self.boundary = boundary
        self.integrator = IntegratorArbitraryGrid(self)

    def levelToNumPoints(self, levelvec):
        return [2 ** levelvec[d] + 1 - int(self.boundary == False) * (int(self.start[d] == 0) + int(self.end[d] == 1))
                for d in range(self.dim)]

    def setCurrentArea(self, start, end, levelvec):
        self.start = start
        self.end = end
        self.levelvec = levelvec
        self.dim = len(self.levelvec)
        self.numPoints = self.levelToNumPoints(levelvec)
        self.numPointsWithBoundary = list(self.numPoints)
        self.length = np.array(end) - np.array(start)
        self.lowerBorder = np.zeros(self.dim, dtype=int)
        self.upperBorder = np.array(self.numPoints, dtype=int)
        if (self.boundary == False):
            for i in range(self.dim):
                if start[i] == self.a[i]:
                    self.numPointsWithBoundary[i] += 1
                    self.lowerBorder[i] = 1
                if end[i] == self.b[i]:
                    self.numPointsWithBoundary[i] += 1
                    self.upperBorder[i] = self.numPoints[i] - 1
                else:
                    self.upperBorder[i] = self.numPoints[i]

        self.coordinate_array = []
        for d in range(self.dim):
            coordinates = np.empty(self.numPoints[d])
            for i in range(self.numPoints[d]):
                coordinates[i] = self.start[d] + (
                    1 - math.cos(math.pi * (i + self.lowerBorder[d]) / (self.numPointsWithBoundary[d] - 1))) * \
               self.length[d] / 2
            self.coordinate_array.append(coordinates)

    def getPoints(self):
        return list(zip(*[g.ravel() for g in np.meshgrid(
            *[[self.get1DCoordinate(p, i) for p in range(self.numPoints[i])] for i in range(self.dim)])]))

    def get1DCoordinate(self, index, d):
        return self.start[d] + (
                    1 - math.cos(math.pi * (index + self.lowerBorder[d]) / (self.numPointsWithBoundary[d] - 1))) * \
               self.length[d] / 2

    def getCoordinate(self, indexvector):
        position = np.zeros(self.dim)
        for d in range(self.dim):
            position[d] = self.get1DCoordinate(indexvector[d], d)
        return position

    def getWeight(self, indexvec):
        weight = 1
        indexvector = list(indexvec)
        for d in range(self.dim):
            indexvector[d] += self.lowerBorder[d]
        for d in range(self.dim):
            if (self.numPointsWithBoundary[d] == 2):
                weight_index_d = 1
            else:
                if indexvector[d] == 0 or indexvector[d] == self.numPointsWithBoundary[d] - 1:
                    weight_index_d = 1.0 / ((self.numPointsWithBoundary[d] - 2) * self.numPointsWithBoundary[d])
                else:
                    weight_index_d = 0.0
                    for j in range(1, math.floor((self.numPointsWithBoundary[d] - 1) / 2.0) + 1):
                        term = 1.0 / (1.0 - 4 * j * j) * math.cos(
                            2 * math.pi * indexvector[d] * j / (self.numPointsWithBoundary[d] - 1))
                        if j == math.floor((self.numPointsWithBoundary[d] - 1) / 2.0):
                            term *= 0.5
                        weight_index_d += term
                    weight_index_d = 2.0 / (self.numPointsWithBoundary[d] - 1) * (1 + 2 * weight_index_d)
            weight *= weight_index_d * self.length[d] / 2
            # print(indexvector,weight_index_d* self.length[d]/2,weight,self.numPoints)
        return weight


# this class generates a grid according to the roots of Chebyshev points and applies a Clenshaw Curtis quadrature
# the formulas are taken from: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.33.3141&rep=rep1&type=pdf
class EquidistantGridGlobal(Grid):
    def __init__(self, a, b, boundary=True):
        self.boundary = boundary
        self.integrator = IntegratorArbitraryGrid(self)
        self.a = a
        self.b = b
        self.dim = len(a)
        self.length = np.array(b) - np.array(a)
        if (self.boundary == False):
            self.lowerBorder = np.ones(self.dim, dtype=int)
        else:
            self.lowerBorder = np.zeros(self.dim, dtype=int)

    def isGlobal(self):
        return True

    def setRefinement(self, refinement, levelvec):
        self.refinement = refinement
        self.coords = []
        self.weights = []
        self.numPoints = np.zeros(self.dim, dtype=int)
        for d in range(self.dim):
            refinementDim = refinement.get_refinement_container_for_dim(d)
            coordsD = self.mapPoints(
                refinementDim.getPointsLine()[self.lowerBorder[d]: -1 * int(self.boundary == False)], self.levelvec[d],
                d)
            weightsD = self.compute_1D_quad_weights(coords1D)
            self.coords.append(coordsD)
            self.weights.append(weightsD)
            self.numPoints[d] = len(coordsD)
        self.numPointsWithoutCoarsening = levelToNumPoints(levelvec)

    def compute_1D_quad_weights(self, grid_1D):
        N = len(grid_1D)
        V = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                V[i, j] = eval_sh_legendre(j, grid_1D[i]) * np.sqrt(2 * j + 1)

        weights = np.linalg.inv(V)[0, :]

        return weights

    def levelToNumPoints(self, levelvec):
        if hasattr(self, 'numPoints'):
            return self.numPoints
        else:
            return [2 ** levelvec[d] + 1 - int(self.boundary == False) * 2 for d in range(self.dim)]

    def getPoints(self):
        return list(zip(*[g.ravel() for g in np.meshgrid(*self.coords)]))

    def getCoordinate(self, indexvector):
        position = np.zeros(self.dim)
        for d in range(self.dim):
            position[d] = self.coords[d][indexvector[d]]
        return position

    def getWeight(self, indexvector):
        weight = 1
        for d in range(self.dim):
            weight *= self.weights[d][indexvector[d]]
        return weight

    def mapPoints(self, equidistantAdaptivePoints, level, d):
        return equidistantAdaptivePoints


# this class generates a grid according to the roots of Chebyshev points and applies a Clenshaw Curtis quadrature
# the formulas are taken from: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.33.3141&rep=rep1&type=pdf
class ClenshawCurtisGridGlobal(EquidistantGridGlobal):
    def __init__(self, a, b, boundary=True):
        self.boundary = boundary
        self.integrator = IntegratorArbitraryGrid(self)
        self.a = a
        self.b = b
        self.dim = len(a)
        self.length = np.array(b) - np.array(a)
        if (self.boundary == False):
            self.lowerBorder = np.ones(self.dim, dtype=int)
        else:
            self.lowerBorder = np.zeros(self.dim, dtype=int)

    '''
    def setCurrentArea(self,start,end,levelvec):
        self.start = start
        self.end = end
        self.levelvec = levelvec
        self.dim = len(self.levelvec)
        self.numPoints = self.levelToNumPoints(levelvec)
        self.numPointsWithBoundary = list(self.numPoints)
        self.length = np.array(end) - np.array(start)
        self.lowerBorder = np.zeros(self.dim,dtype=int)
        self.upperBorder = np.array(self.numPoints, dtype=int)
        if(self.boundary == False):
            for i in range(self.dim):
                if start[i]==0 :
                    self.numPointsWithBoundary[i] += 1
                    self.lowerBorder[i] = 1
                if end[i] == 1:
                    self.numPointsWithBoundary[i] += 1
                    self.upperBorder[i] = self.numPoints[i]-1
                else:
                    self.upperBorder[i] = self.numPoints[i]
    '''

    def mapPoints(self, equidistantAdaptivePoints, level, d):
        coords = np.zeros(len(equidistantAdaptivePoints))
        for i, p in enumerate(equidistantAdaptivePoints):
            index = self.getIndexTo1DCoordinate(p, level)
            coords[i] = self.get1DCoordinate(index, d)
        return coords

    def get1DCoordinate(self, index, d):
        return self.a[d] + (
                    1 - math.cos(math.pi * (index + self.lowerBorder[d]) / (self.numPointsWithoutCoarsening[d] - 1))) * \
               self.length[d] / 2


import numpy.polynomial.legendre as legendre


# this class generates a grid according to the Gauss-Legendre quadrature
class GaussLegendreGrid(Grid):
    def __init__(self, boundary=True):
        self.boundary = True  # never points on boundary
        self.integrator = IntegratorArbitraryGrid(self)

    def levelToNumPoints(self, levelvec):
        return [2 ** levelvec[d] for d in range(self.dim)]

    # Gauss Legendre grids are not nested!
    def isNested(self):
        return False

    def setCurrentArea(self, start, end, levelvec):
        self.start = np.array(start)
        self.end = np.array(end)
        self.length = np.array(end) - np.array(start)
        self.levelvec = levelvec
        self.dim = len(levelvec)
        self.numPoints = self.levelToNumPoints(levelvec)
        self.coordinate_array = []
        self.weights = []
        # prepare coordinates and weights
        for d in range(self.dim):
            coordsD, weightsD = legendre.leggauss(int(self.numPoints[d]))
            coordsD = np.array(coordsD)
            coordsD += np.ones(int(self.numPoints[d]))
            coordsD *= self.length[d] / 2.0
            coordsD += self.start[d]
            weightsD = np.array(weightsD) * self.length[d] / 2
            self.coordinate_array.append(coordsD)
            self.weights.append(weightsD)

    def getPoints(self):
        return list(zip(*[g.ravel() for g in np.meshgrid(*self.coordinate_array)]))

    def getWeight(self, indexvector):
        weight = 1
        for d in range(self.dim):
            weight *= self.weights[d][indexvector[d]]
        return weight

    def get_weights(self):
        return list(self.getWeight(index) for index in zip(*[g.ravel() for g in np.meshgrid(*[range(self.numPoints[d]) for d in range(self.dim)])]))
