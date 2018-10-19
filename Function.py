import abc, logging
import numpy as np
import scipy.special
import math
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# The function class is used to define several functions for testing the algorithm
# it defines the basic interface that is used by the algorithm
class Function(object):
    # initialization if necessary
    def __init__(self):
        self.log = logging.getLogger(__name__)

    # evaluates the function at the specified coordinate
    @abc.abstractmethod
    def eval(self, coordinates):
        return

    # this returns the analytic solution of the integral in the specified area
    # currently necessary for the error estimator
    @abc.abstractmethod
    def getAnalyticSolutionIntegral(self, start, end):
        return

    # this method plots the function in the specified area for 2D
    def plot(self, start, end, filename=None):
        dim = len(start)
        if dim > 2:
            print("Cannot plot function with dim > 2")
            return
        xArray = np.linspace(start[0], end[0], 10 ** 2)
        yArray = np.linspace(start[1], end[1], 10 ** 2)
        X = [x for x in xArray]
        Y = [y for y in yArray]
        X, Y = np.meshgrid(X, Y)
        Z = np.zeros(np.shape(X))
        for i in range(len(X)):
            for j in range(len(X[i])):
                # print(X[i,j],Y[i,j],self.eval((X[i,j],Y[i,j])))
                Z[i, j] = self.eval((X[i, j], Y[i, j]))
        # Z=self.eval((X,Y))
        # print Z
        fig = plt.figure(figsize=(14, 6))

        # `ax` is a 3D-aware axis instance, because of the projection='3d' keyword argument to add_subplot
        ax = fig.add_subplot(1, 2, 1, projection='3d')

        # p = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        p = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False)
        # plt.show()
        if filename is not None:
            fig.savefig(filename, bbox_inches='tight')
        plt.show()


# This class composes different functions that fullfill the function interface to generate a composition of functions
# Each Function can be weighted by a factor to allow for a flexible composition
class FunctionCompose(Function):
    def __init__(self, functions):
        self.functions = functions

    def eval(self, coordinates):
        result = 0.0
        for (f, factor) in self.functions:
            result += f.eval(coordinates) * factor
        return result

    def getAnalyticSolutionIntegral(self, start, end):
        result = 0.0
        for (f, factor) in self.functions:
            result += f.getAnalyticSolutionIntegral(start, end) * factor
        return result


# This Function represents the corner Peak f the genz test functions
class GenzCornerPeak(Function):
    def __init__(self, coeffs):
        self.coeffs = coeffs
        self.dim = len(coeffs)

    def eval(self, coordinates):
        result = 1
        for d in range(self.dim):
            result += self.coeffs[d] * coordinates[d]
        return result ** (-self.dim - 1)


class GenzProductPeak(Function):
    def __init__(self, coeffs, midPoint):
        self.coeffs = coeffs
        self.midPoint = midPoint
        self.dim = len(coeffs)
        self.factor = 10 ** (-self.dim)

    def eval(self, coordinates):
        result = 1
        for d in range(self.dim):
            result /= (self.coeffs[d] ** (-2) + (coordinates[d] - self.midPoint[d]) ** 2)
        return result * self.factor

    def getAnalyticSolutionIntegral(self, start, end):
        result = 1
        for d in range(self.dim):
            result *= np.arctan(self.coeffs[d] * (self.midPoint[d] - start[d])) * self.coeffs[d] - np.arctan(
                self.coeffs[d] * (self.midPoint[d] - end[d])) * self.coeffs[d]
        return result * self.factor


class GenzOszillatory(Function):
    def __init__(self, coeffs, offset):
        self.coeffs = coeffs
        self.dim = len(coeffs)
        self.offset = offset

    def eval(self, coordinates):
        result = 2 * math.pi * self.offset
        for d in range(self.dim):
            result += self.coeffs[d] * coordinates[d]
        return math.cos(result)


class GenzDiscontinious(Function):
    def __init__(self, coeffs, border):
        self.coeffs = coeffs
        self.border = border
        self.dim = len(coeffs)

    def eval(self, coordinates):
        result = 0
        for d in range(self.dim):
            if coordinates[d] >= self.border[d]:
                return 0.0
            result -= self.coeffs[d] * coordinates[d]
        return np.exp(result)

    def getAnalyticSolutionIntegral(self, start, end):
        result = 1
        for d in range(self.dim):
            if start[d] >= self.border[d]:
                return 0.0
            else:
                end[d] = min(end[d], self.border[d])
                result *= (np.exp(-self.coeffs[d] * start[d]) - np.exp(-self.coeffs[d] * end[d])) / self.coeffs[d]
        return result


class GenzC0(Function):
    def __init__(self, coeffs, midPoint):
        self.coeffs = coeffs
        self.midPoint = midPoint
        self.dim = len(coeffs)

    def eval(self, coordinates):
        result = 0
        for d in range(self.dim):
            result -= self.coeffs[d] * abs(coordinates[d] - self.midPoint[d])
        return np.exp(result)

    def getAnalyticSolutionIntegral(self, start, end):
        result = 1
        for d in range(self.dim):
            one_d_integral = 0
            if start[d] < self.midPoint[d]:
                one_d_integral += (np.exp(self.coeffs[d] * (start[d] - self.midPoint[d])) - 1) / self.coeffs[d]
            if end[d] > self.midPoint[d]:
                one_d_integral += (np.exp(self.coeffs[d] * (end[d] - self.midPoint[d])) - 1) / self.coeffs[d]
            result *= one_d_integral
        return result


# This function is the test case function 2 of the paper from Jakeman and Roberts: https://arxiv.org/pdf/1110.0010.pdf
class Function2Jakeman(Function):
    def __init__(self, midpoints, coefficients):
        self.midpoints = midpoints
        self.coefficients = coefficients

    def eval(self, coordinates):
        dim = len(coordinates)
        assert (dim == len(self.coefficients))
        summation = 0.0
        for d in range(dim):
            summation -= self.coefficients[d] * (coordinates[d] - self.midpoints[d]) ** 2
        return np.exp(summation)

    def getAnalyticSolutionIntegral(self, start, end):
        dim = len(start)
        # print lowerBounds,upperBounds,coefficients, midpoints
        result = 1.0
        sqPiHalve = np.sqrt(np.pi) * 0.5
        for d in range(dim):
            result = result * (
                        sqPiHalve * scipy.special.erf(np.sqrt(self.coefficients[d]) * (end[d] - self.midpoints[d])) -
                        sqPiHalve * scipy.special.erf(
                    np.sqrt(self.coefficients[d]) * (start[d] - self.midpoints[d]))) / np.sqrt(self.coefficients[d])
        return result


# This function is the first test function of the paper from Gerstner and Griebel:
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.33.3141&rep=rep1&type=pdf
class FunctionGriebel(Function):
    def eval(self, coordinates):
        dim = len(coordinates)
        prod = 1.0
        for d in range(dim):
            prod *= coordinates[d] ** (1.0 / dim)
        return (1 + 1.0 / dim) ** dim * prod

    def getAnalyticSolutionIntegral(self, start, end):
        dim = len(start)
        result = 1.0
        for d in range(dim):
            result = result * (
                        end[d] ** (1 + 1.0 / dim) / (1 + 1.0 / dim) - start[d] ** (1 + 1.0 / dim) / (1 + 1.0 / dim))
        return (1 + 1.0 / dim) ** dim * result


# This is a variant of the generalized normal distribution
# Currently the analytic solution is not correct!!!
class FunctionGeneralizedNormal(Function):
    def __init__(self, midpoints, coefficients, exp):
        self.midpoints = midpoints
        self.coefficients = coefficients
        self.exponent = exp

    def eval(self, coordinates):
        dim = len(coordinates)
        assert (dim == len(self.coefficients))
        summation = 0.0
        for d in range(dim):
            summation -= self.coefficients[d] * (abs(coordinates[d] - self.midpoints[d])) ** 2
        return np.exp(summation ** self.exponent)

    def getAnalyticSolutionIntegral(self, start, end):
        dim = len(start)
        # print lowerBounds,upperBounds,coefficients, midpoints
        result = 1.0
        sqPiHalve = np.sqrt(np.pi) * 0.5
        for d in range(dim):
            result = result * (
                        sqPiHalve * scipy.special.erf(np.sqrt(self.coefficients[d]) * (end[d] - self.midpoints[d])) -
                        sqPiHalve * scipy.special.erf(
                    np.sqrt(self.coefficients[d]) * (start[d] - self.midpoints[d]))) / np.sqrt(self.coefficients[d])
        return result
