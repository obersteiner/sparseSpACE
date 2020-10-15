import abc, logging
import numpy as np
import scipy.special
import math
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Mapping, MutableMapping, Sequence, Iterable, List, Set, Tuple, Union

# The function class is used to define several functions for testing the algorithm
# it defines the basic interface that is used by the algorithm
class Function(object):
    # initialization if necessary
    def __init__(self):
        self.log = logging.getLogger(__name__)
        self.f_dict = {}
        self.old_f_dict = {}
        self.do_cache = True  # indicates whether function values should be cached
        self.debug = False

    def reset_dictionary(self) -> None:
        # self.old_f_dict = {**self.old_f_dict, **self.f_dict}
        self.old_f_dict = {}
        self.f_dict = {}

    def __call__(self, coordinates: Union[Tuple[float, ...], Sequence[Tuple[float]]]) -> Sequence[float]:
        f_value = None
        if np.isscalar(coordinates[0]):
            # single evaluation point
            if self.do_cache:
                coords = tuple(coordinates)
                f_value = self.f_dict.get(coords, None)
                if f_value is None:
                    f_value = self.old_f_dict.get(coords, None)
                    if f_value is not None:
                        self.f_dict[coords] = f_value
            if f_value is None:
                f_value = self.eval(coords)
                if self.do_cache:
                    self.f_dict[coords] = f_value
            if np.isscalar(f_value):
                f_value = [f_value]
            assert len(f_value) == self.output_length(), "Wrong output_length()! Adjust the output length in your function!"
            return np.array(f_value)
        else:
            if not isinstance(coordinates[0], tuple):
                print("Warning: not passing tuples to Function -> less efficient!")
                coordinates = [tuple(c) for c in coordinates]
            f_values = np.asarray(self.eval_vectorized(np.asarray(coordinates)))
            f_values = f_values.reshape((len(coordinates), self.output_length()))
            self.f_dict.update(zip(coordinates, f_values))
            return f_values


    def eval_vectorized(self, coordinates: Sequence[Sequence[float]]):
        f_values = np.empty((*np.shape(coordinates)[:-1], self.output_length()))
        for i, coordinate in enumerate(coordinates):
            if np.isscalar(coordinate[0]):
                f_values[i, :] = self.eval(coordinate)
            else:
                f_values[i, :] = self.eval_vectorized(coordinate)
        return f_values

    def check_vectorization(self, coordinates, result):
        if self.debug:
            for i in range(len(coordinates)):
                if not math.isclose(result[i], self.eval(coordinates[i])):
                    print(result[i], self.eval(coordinates[i]), coordinates[i])
                assert math.isclose(result[i], self.eval(coordinates[i]))

    def deactivate_caching(self) -> None:
        self.do_cache = False

    def get_f_dict_size(self) -> int:
        return len(self.f_dict)

    def get_f_dict_points(self):
        return list(self.f_dict.keys())

    def get_f_dict_values(self):
        return list(self.f_dict.values())

    # evaluates the function at the specified coordinate
    @abc.abstractmethod
    def eval(self, coordinates: Tuple[float, ...]) -> Sequence[float]:
        pass

    # this returns the analytic solution of the integral in the specified area
    # currently necessary for the error estimator
    def getAnalyticSolutionIntegral(self, start: Sequence[float], end: Sequence[float]) -> Sequence[float]:
        self.dim = len(start)
        if self.dim == 3:
            f = lambda x, y, z: self.eval([x, y, z])
            return \
                integrate.tplquad(f, start[2], end[2], lambda x: start[1], lambda x: end[1], lambda x, y: start[0],
                                  lambda x, y: end[0])[0]
        elif self.dim == 2:
            f = lambda x, y: self.eval([x, y])
            return integrate.dblquad(f, start[1], end[1], lambda x: start[0], lambda x: end[0])[0]
        else:
            assert False

    # this method plots the function in the specified area for 2D
    def plot(self, start: Sequence[float], end: Sequence[float], filename: str=None, plotdimension: int=0, dpi: int=100, width: float=14, height: float=6, points_per_dim=100, plotdimensions=None, show_plot=True) -> None:
        dim = len(start)
        if dim > 2:
            print("Cannot plot function with dim > 2")
            return
        xArray = np.linspace(start[0], end[0], points_per_dim)
        yArray = np.linspace(start[1], end[1], points_per_dim)
        X = [x for x in xArray]
        Y = [y for y in yArray]
        X, Y = np.meshgrid(X, Y)
        evals = np.zeros(np.shape(X) + (self.output_length(),))
        for i in range(len(X)):
            for j in range(len(X[i])):
                evals[i, j] = self.__call__((X[i, j], Y[i, j]))
        if plotdimensions is None:
            plotdimensions = [plotdimension]
        single_dim = len(plotdimensions) == 1
        consistent_axes = False
        if consistent_axes:
            assert not single_dim
            # Find the minimum and maximum output value to be able to set a
            # consistent z axis
            minz = evals[0,0,plotdimensions[0]]
            maxz = minz
            for output_dim in plotdimensions:
                minz = min(minz, evals.T[output_dim].min())
                maxz = max(maxz, evals.T[output_dim].max())
        for output_dim in plotdimensions:
            Z = np.zeros(np.shape(X))
            for i in range(len(X)):
                for j in range(len(X[i])):
                    Z[i, j] = self.__call__((X[i, j], Y[i, j]))[output_dim]
            # ~ fig = plt.figure(figsize=(14, 6))
            fig = plt.figure(figsize=(21, 9))

            # `ax` is a 3D-aware axis instance, because of the projection='3d' keyword argument to add_subplot
            ax = fig.add_subplot(1, 2, 1, projection='3d')
            if consistent_axes:
                ax.set_zlim(minz, maxz)

            # ~ p = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            p = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False, vmin=100)
            if filename is not None:
                if single_dim:
                    fig.savefig(filename, bbox_inches='tight')
                else:
                    fig.savefig(f"{filename}_{output_dim}", bbox_inches='tight')
            if show_plot:
                plt.show()

    def output_length(self) -> int:
        return 1


from scipy import integrate
from scipy.stats import norm


# This simple function can be used for test purposes
class ConstantValue(Function):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def eval(self, coordinates):
        return self.value

    def getAnalyticSolutionIntegral(self, start, end):
        dim = len(start)
        integral = 1.0
        for d in range(dim):
            integral *= end[d] - start[d]
        integral *= self.value

class FunctionDiagonalDiscont(Function):
    def eval(self, coordinates):
        dim = len(coordinates)
        if sum(coordinates) < 1:
            return np.ones(1)
        else:
            return np.zeros(1)

    def getAnalyticSolutionIntegral(self, start, end):
        dim = len(start)
        for d in range(dim):
            assert start[d] == 0
            assert end[d] == 1
        return 1/math.factorial(dim)

class FunctionShift(Function):
    def __init__(self, function, shift):
        super().__init__()
        self.function = function
        self.shift = shift  # a list of functions that can manipulate the coordinates for every dimension

    def eval(self, coordinates):
        shifted_coordinates = self.shift(coordinates)
        return self.function.eval(shifted_coordinates)

    def getAnalyticSolutionIntegral(self, start, end):
        start_shifted = self.shift(start)
        end_shifted = self.shift(end)
        return self.function.getAnalyticSolutionIntegral(start_shifted, end_shifted)


class FunctionUQNormal(Function):
    def __init__(self, function, mean, std_dev, a, b):
        super().__init__()
        self.dim = len(mean)
        self.mean = mean
        self.std_dev = std_dev
        self.function = function
        self.a_global = a
        self.b_global = b

    def eval(self, coordinates):
        return self.function(np.asarray(coordinates) * np.asarray(self.std_dev) + np.asarray(self.mean))

    def eval_with_normal(self, coordinates):
        value = self.eval(coordinates)
        dim = len(coordinates)
        # add contribution of normal distribution
        summation = 0
        for d in range(dim):
            summation -= (coordinates[d]) ** 2 / (2 * 1)
        return value * np.exp(summation)

    def getAnalyticSolutionIntegral(self, start, end):
        if self.dim == 3:
            f = lambda x, y, z: self.eval_with_normal([x, y, z])
            normalization = 1
            for d in range(len(start)):
                S = norm.cdf(self.b_global[d]) - norm.cdf(self.a_global[d])
                normalization *= 1.0 / (S * math.sqrt(2 * math.pi * 1))
            return normalization * \
                   integrate.tplquad(f, start[2], end[2], lambda x: start[1], lambda x: end[1], lambda x, y: start[0],
                                     lambda x, y: end[0])[0]
        elif self.dim == 2:
            f = lambda x, y: self.eval_with_normal([x, y])
            normalization = 1
            for d in range(len(start)):
                S = norm.cdf(self.b_global[d]) - norm.cdf(self.a_global[d])
                normalization *= 1.0 / (S * math.sqrt(2 * math.pi * 1))
            return normalization * integrate.dblquad(f, start[1], end[1], lambda x: start[0], lambda x: end[0])[0]


class FunctionUQNormal2(Function):
    def __init__(self, function, mean, std_dev, a, b):
        super().__init__()
        self.mean = mean
        self.std_dev = std_dev
        self.function = function
        self.a_global = a
        self.b_global = b
        self.dim = len(mean)

    def eval(self, coordinates):
        return self.function.eval(coordinates)

    def eval_with_normal(self, coordinates):
        dim = len(coordinates)
        # add contribution of normal distribution
        return np.prod([norm.pdf(x=coordinates[d], loc=self.mean[d], scale=self.std_dev[d]) / (
                    norm.cdf(self.b_global[d], loc=self.mean[d], scale=self.std_dev[d]) - norm.cdf(self.a_global[d],
                                                                                                   loc=self.mean[d],
                                                                                                   scale=self.std_dev[
                                                                                                       d])) for d in
                        range(dim)])

    def getAnalyticSolutionIntegral(self, start, end):
        if self.dim == 3:
            f = lambda x, y, z: self.eval([x, y, z]) * self.eval_with_normal([x, y, z])
            return integrate.tplquad(f, start[2], end[2], lambda x: start[1], lambda x: end[1], lambda x, y: start[0],
                                     lambda x, y: end[0])[0]
        elif self.dim == 2:
            f = lambda x, y: self.eval([x, y]) * self.eval_with_normal([x, y])
            return integrate.dblquad(f, start[1], end[1], lambda x: start[0], lambda x: end[0])[0]
        else:
            assert False


# This function works only with single-dimensional output functions
class FunctionUQWeighted(Function):
    def __init__(self, function, weight_function):
        super().__init__()
        self.function = function
        self.weight_function = weight_function

    def eval(self, coordinates):
        func_output = self.function(coordinates)[0]
        weight_output = self.weight_function(coordinates)[0]
        return func_output * weight_output


# An UQ test function: https://www.sfu.ca/~ssurjano/canti.html
class FunctionCantileverBeamD(Function):
    def __init__(self, width=20.0, thickness=2.0):
        super().__init__()
        self.w = width
        self.t = thickness

    def eval(self, coordinates):
        assert len(coordinates) == 3
        w = self.w
        t = self.t
        L = 100.0
        E, Y, X = coordinates
        D = 4.0 * L ** 3 / (E * w * t) * math.sqrt((Y / t ** 2) ** 2 + (X / w ** 2) ** 2)
        return [D, 1.0]

    def getAnalyticSolutionIntegral(self, start, end): assert "not implemented"

class CustomFunction(Function):
    def __init__(self, function, output_length=1):
        super().__init__()
        self.function = function
        self.output_length_parameter = output_length

    def eval(self, coordinates):
        return self.function(coordinates)

    def output_length(self) -> int:
        return self.output_length_parameter

# g-function of Sobol: https://www.sfu.ca/~ssurjano/gfunc.html
class FunctionG(Function):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.a = 0.5 * np.array(range(dim))

    def eval(self, coordinates):
        assert len(coordinates) == self.dim
        a = self.a
        return np.prod([(abs(4.0 * coordinates[d] - 2.0) + a[d]) / (1.0 + a[d]) for d in range(self.dim)])

    # Uniform distributions in [0, 1] are required for this Function.
    def get_expectation(self): return 1.0

    def get_variance(self):
        mom2 = np.prod([1.0 + 1.0 / (3.0 * (1.0 + a_d) ** 2) for a_d in self.a])
        return mom2 - 1.0

    # ~ def get_first_order_sobol_indices(self):
        # This seems to be wrong
        # ~ fac = 1.0 / np.prod([1.0 / (3.0 * (1.0 + a_d) ** 2) for a_d in self.a])
        # ~ return [fac * 1.0 / (3.0 * (1.0 + self.a[d]) ** 2) for d in range(self.dim)]

        # ~ ivar = 1.0 / self.get_variance()
        # ~ return [ivar * (1.0 + 1.0 / (3 * (1.0 + self.a[i]) ** 2)) for i in range(self.dim)]

    def getAnalyticSolutionIntegral(self, start, end):
        assert all([v == 0.0 for v in start])
        assert all([v == 1.0 for v in end])
        return self.get_expectation()


class FunctionGShifted(FunctionG):
    def eval(self, coordinates):
        assert all([0.0 <= v <= 1.0 for v in coordinates])
        # Shift the coordinates by 0.2 in every dimension
        coords = [v + 0.2 for v in coordinates]
        # Go back to the other side so that expectation and variance do not change
        coords = [v if v <= 1.0 else v - 1.0 for v in coords]
        return super().eval(coords)


class FunctionUQ(Function):
    def eval(self, coordinates):
        assert (len(coordinates) == 3), len(coordinates)
        parameter1 = coordinates[0]
        parameter2 = coordinates[1]
        parameter3 = coordinates[2]

        # Model with discontinuity
        # Nicholas Zabarras Paper: „Sparse grid collocation schemes for stochastic natural convection problems“
        # e^(-x^2 + 2*sign(y))
        value_of_interest = math.exp(-parameter1 ** 2 + 2 * np.sign(parameter2)) + parameter3
        return value_of_interest

    def getAnalyticSolutionIntegral(self, start, end):
        f = lambda x, y, z: self.eval([x, y, z])
        return integrate.tplquad(f, start[2], end[2], lambda x: start[1], lambda x: end[1], lambda x, y: start[0],
                                 lambda x, y: end[0])[0]


class FunctionUQShifted(FunctionUQ):
    def eval(self, coordinates):
        # Move the discontinuity away from the middle
        coords = np.array([coordinates[0], coordinates[1] + 0.221413, coordinates[2]])
        return super().eval(coords)


from scipy.stats import truncnorm


class FunctionUQ2(Function):
    def eval(self, coordinates):
        # print(coordinates)
        assert (len(coordinates) == 2)
        parameter1 = coordinates[0]
        parameter2 = coordinates[1]

        # Model with discontinuity
        # Nicholas Zabarras Paper: „Sparse grid collocation schemes for stochastic natural convection problems“
        # e^(-x^2 + 2*sign(y))
        value_of_interest = math.exp(-parameter1 ** 2 + 2 * np.sign(parameter2))
        return value_of_interest

    def getAnalyticSolutionIntegral(self, start, end):
        f = lambda x, y: self.eval([x, y])
        return integrate.dblquad(f, start[1], end[1], lambda x: start[0],
                                 lambda x: end[0])[0]


# This class composes different functions that fullfill the function interface to generate a composition of functions
# Each Function can be weighted by a factor to allow for a flexible composition
class FunctionCompose(Function):
    def __init__(self, functions):
        super().__init__()
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


class FunctionLinear(Function):
    def __init__(self, coeffs):
        super().__init__()
        self.coeffs = np.array(coeffs)
        self.dim = len(coeffs)

    def eval(self, coordinates):
        result = 1
        for d in range(self.dim):
            result *= self.coeffs[d] * coordinates[d]
        return result

    def eval_vectorized(self, coordinates: Sequence[Sequence[float]]):
        result = np.prod(coordinates * self.coeffs,axis=-1)
        #for i in range(len(coordinates)):
        #    print(result[i], self.eval(coordinates[i]))
        #    assert result[i] == self.eval(coordinates[i])
        return result

    def getAnalyticSolutionIntegral(self, start, end):
        result = 1.0
        for d in range(self.dim):
            result *= self.coeffs[d] * (end[d]**2/2 - start[d]**2/2)
        return result


class FunctionMultilinear(Function):
    def __init__(self, coeffs):
        super().__init__()
        self.coeffs = np.array(coeffs)
        self.dim = len(coeffs)

    def eval(self, coordinates):
        result = 0.0
        for d in range(self.dim):
            result += self.coeffs[d] * coordinates[d]
        return result

    def getAnalyticSolutionIntegral(self, start, end):
        result = 0.0
        for d in range(self.dim):
            result += self.coeffs[d] * (end[d]**2/2 - start[d]**2/2)
        return result


# This can be used when calculating the variance
class FunctionPower(Function):
    def __init__(self, function, exponent):
        super().__init__()
        self.function = function
        self.exponent = exponent

    def eval(self, coordinates):
        val_f = self.function(coordinates)
        return [v ** self.exponent for v in val_f]

    def getAnalyticSolutionIntegral(self, start, end): assert "Not implemented"

    def output_length(self): return self.function.output_length()


# This can be used when calculating the PCE
class FunctionPolysPCE(Function):
    def __init__(self, function, polys, norms):
        super().__init__()
        self.function = function
        self.polys = polys
        self.norms = norms
        self.output_dimension = function.output_length() * len(polys)

    def eval(self, coordinates):
        values = []
        val_f = self.function(coordinates)
        for i in range(len(self.polys)):
            val_poly = self.polys[i](*coordinates) / self.norms[i]
            # Concatenation required for functions with multidimensional output
            values += [v * val_poly for v in val_f]
        return values

    def getAnalyticSolutionIntegral(self, start, end): assert "Not implemented"

    def output_length(self): return self.output_dimension


class FunctionInverseTransform(Function):
    def __init__(self, function, distributions):
        super().__init__()
        self.function = function
        self.ppfs = [dist.ppf for dist in distributions]

    def eval(self, coords_transformed):
        assert all([0 <= v <= 1 for v in coords_transformed]), "PPF functions require the points to be in [0,1]"
        coordinates = [ppf(coords_transformed[d]) for d,ppf in enumerate(self.ppfs)]
        assert not any([math.isinf(v) for v in coordinates]), "infinite coordinates, maybe boundary needs to be set to true in a Grid"
        return self.function(coordinates)

    def getAnalyticSolutionIntegral(self, start, end): assert "Not implemented"

    def output_length(self): return self.function.output_length()


class FunctionCustom(Function):
    def __init__(self, func, output_dim=None):
        super().__init__()
        self.func = func
        self.has_multiple_functions = hasattr(self.func, "__iter__")
        self.output_dimension = output_dim
        if self.output_dimension is None:
            self.output_dimension = len(self.func) if self.has_multiple_functions else 1

    def eval(self, coordinates):
        if self.has_multiple_functions:
            result = [float(f(coordinates)) for f in self.func]
        else:
            result = self.func(coordinates)
        return result

    def output_length(self): return self.output_dimension


class FunctionConcatenate(Function):
    def __init__(self, funcs):
        super().__init__()
        self.funcs = funcs
        self.output_dimension = sum([f.output_length() for f in funcs])

    def eval(self, coordinates):
        return np.concatenate([f(coordinates) for f in self.funcs])

    def getAnalyticSolutionIntegral(self, start, end): assert "Not available"

    def output_length(self): return self.output_dimension


class FunctionPolynomial(Function):
    def __init__(self, coeffs, degree=2):
        super().__init__()
        self.coeffs = np.array(coeffs)
        self.dim = len(coeffs)
        self.degree = degree

    def eval(self, coordinates):
        result = 1
        for d in range(self.dim):
            result *= self.coeffs[d] * coordinates[d] ** self.degree
        return result

    def getAnalyticSolutionIntegral(self, start, end):
        result = 1.0
        for d in range(self.dim):
            result *= self.coeffs[d] * (end[d]**(self.degree+1)/(self.degree + 1) - start[d]**(self.degree+1)/(self.degree + 1))
        return result


class LambdaFunction(Function):
    def __init__(self, function, anti_derivative):
        super().__init__()
        self.function = function
        self.anti_derivative = anti_derivative

    def eval(self, coordinates):
        return self.function(coordinates)

    def getAnalyticSolutionIntegral(self, start, end):
        anti_derivative_value_left = self.anti_derivative(start)
        anti_derivative_value_right = self.anti_derivative(end)

        return anti_derivative_value_right - anti_derivative_value_left


class Polynomial1d(Function):
    """
    This class stores generic polynomials of the form
    p(x) = c_0 * x^0 + c_1 * x^1 +  ... + c_(n-1) * x^(n-1) + c_n * x^n
    """
    def __init__(self, coefficients):
        super().__init__()
        self.coefficients = coefficients

        self.anti_derivative_coefficients = [0.0] * (len(coefficients) + 1)

        # c_0 is 0 by default
        for i in range(1, len(self.coefficients) + 1):
            self.anti_derivative_coefficients[i] = self.coefficients[i - 1] * 1 / i

    def getAnalyticSolutionIntegral(self, start: Sequence[float], end: Sequence[float]):
        return self.eval_anti_derivative(end[0]) - self.eval_anti_derivative(start[0])

    def eval(self, coordinates: Tuple[float, ...]):
        value = 0

        for i in range(len(self.coefficients)):
            value += self.coefficients[i] * (coordinates[0] ** i)

        return value

    def eval_anti_derivative(self, coordinates: float):
        value = 0

        for i in range(len(self.anti_derivative_coefficients)):
            value += self.anti_derivative_coefficients[i] * (coordinates ** i)

        return value


# This Function represents the corner Peak f the genz test functions
class GenzCornerPeak(Function):
    def __init__(self, coeffs):
        super().__init__()
        self.coeffs = np.array(coeffs)
        self.dim = len(coeffs)

    def eval(self, coordinates):
        result = 1
        for d in range(self.dim):
            result += self.coeffs[d] * coordinates[d]
        return result ** (-self.dim - 1)

    def eval_vectorized(self, coordinates: Sequence[Sequence[float]]):
        result = 1 + np.inner(coordinates, self.coeffs)
        result = result ** (-self.dim - 1)
        self.check_vectorization(coordinates, result)
        return result

    def getAnalyticSolutionIntegral(self, start, end):
        factor = ((-1) ** self.dim) * 1.0 / (math.factorial(self.dim) * np.prod(self.coeffs))
        combinations = list(zip(*[g.ravel() for g in np.meshgrid(*[[0, 1] for d in range(self.dim)])]))
        result = 0
        for c in combinations:
            partial_result = 1
            for d in range(self.dim):
                if c[d] == 1:
                    value = start[d]
                else:
                    value = end[d]
                partial_result += value * self.coeffs[d]
            result += (-1) ** sum(c) * partial_result ** -1
        return factor * result


class GenzProductPeak(Function):
    def __init__(self, coefficients, midpoint):
        super().__init__()
        self.coeffs = np.asarray(coefficients)
        self.midPoint = np.asarray(midpoint)
        self.dim = len(coefficients)
        self.factor = 10 ** (-self.dim)

    def eval(self, coordinates):
        result = self.factor
        for d in range(self.dim):
            result /= (self.coeffs[d] ** (-2) + (coordinates[d] - self.midPoint[d]) ** 2)
        return result

    def eval_vectorized(self, coordinates: Sequence[Sequence[float]]):
        result = np.prod(self.coeffs ** (-2) + (coordinates - self.midPoint) ** (2), axis=-1)
        result = self.factor / result
        self.check_vectorization(coordinates, result)
        return result

    def getAnalyticSolutionIntegral(self, start, end):
        result = 1
        for d in range(self.dim):
            result *= np.arctan(self.coeffs[d] * (self.midPoint[d] - start[d])) * self.coeffs[d] - np.arctan(
                self.coeffs[d] * (self.midPoint[d] - end[d])) * self.coeffs[d]
        return result * self.factor


class GenzOszillatory(Function):
    def __init__(self, coeffs, offset):
        super().__init__()
        self.coeffs = coeffs
        self.dim = len(coeffs)
        self.offset = offset

    def eval(self, coordinates):
        result = 2 * math.pi * self.offset
        for d in range(self.dim):
            result += self.coeffs[d] * coordinates[d]
        return math.cos(result)

    def eval_vectorized(self, coordinates: Sequence[Sequence[float]]):
        result = 2 * math.pi * self.offset + np.inner(coordinates, self.coeffs)
        result = np.cos(result)
        self.check_vectorization(coordinates, result)
        return result

    def getAnalyticSolutionIntegral(self, start, end):
        not_zero_dims = [d for d in range(self.dim) if self.coeffs[d] != 0]
        zero_dims = [d for d in range(self.dim) if self.coeffs[d] == 0]
        factor_zero_dims = np.prod([end[d] - start[d] for d in zero_dims])
        factor = ((-1) ** int(math.floor(len(not_zero_dims) / 2))) * 1.0 / np.prod([self.coeffs[d] for d in not_zero_dims])
        combinations = list(zip(*[g.ravel() for g in np.meshgrid(*[[0, 1] for d in not_zero_dims])]))
        result = 0
        for c in combinations:
            partial_result = 2 * math.pi * self.offset
            for i in range(len(not_zero_dims)):
                d = not_zero_dims[i]
                if c[i] == 1:
                    value = start[d]
                else:
                    value = end[d]
                partial_result += value * self.coeffs[d]
            if len(not_zero_dims) % 2 == 1:
                result += (-1) ** sum(c) * math.sin(partial_result)
            else:
                result += (-1) ** sum(c) * math.cos(partial_result)
        return factor * result * factor_zero_dims


class GenzDiscontinious(Function):
    def __init__(self, coeffs, border):
        super().__init__()
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

    def eval_vectorized(self, coordinates: Sequence[Sequence[float]]):
        result = np.zeros(np.shape(coordinates)[:-1])
        filter = np.all(coordinates < self.border, axis=-1)
        result[filter] = np.exp(-1 * np.inner(coordinates[filter], self.coeffs))
        self.check_vectorization(coordinates, result)
        return result

    def getAnalyticSolutionIntegral(self, start, end):
        result = 1
        end = list(end)
        for d in range(self.dim):
            if start[d] >= self.border[d]:
                return 0.0
            else:
                end[d] = min(end[d], self.border[d])
                result *= (np.exp(-self.coeffs[d] * start[d]) - np.exp(-self.coeffs[d] * end[d])) / self.coeffs[d]
        return result

class GenzDiscontinious2(Function):
    def __init__(self, coeffs, border):
        super().__init__()
        self.coeffs = coeffs
        self.border = border
        self.dim = len(coeffs)

    def eval(self, coordinates):
        result = 0
        for d in range(self.dim):
            if coordinates[d] >= self.border[d]:
                return [0.0, 0.0]
            result -= self.coeffs[d] * coordinates[d]
        return [np.exp(result), np.exp(result)]

    def getAnalyticSolutionIntegral(self, start, end):
        result = 1
        end = list(end)
        for d in range(self.dim):
            if start[d] >= self.border[d]:
                return 0.0
            else:
                end[d] = min(end[d], self.border[d])
                result *= (np.exp(-self.coeffs[d] * start[d]) - np.exp(-self.coeffs[d] * end[d])) / self.coeffs[d]
        return np.asarray([result, result])

class GenzC0(Function):
    def __init__(self, coeffs, midpoint):
        super().__init__()
        self.coeffs = coeffs
        self.midPoint = midpoint
        self.dim = len(coeffs)

    def eval(self, coordinates):
        result = 0
        for d in range(self.dim):
            result -= self.coeffs[d] * abs(coordinates[d] - self.midPoint[d])
        return np.exp(result)

    def eval_vectorized(self, coordinates: Sequence[Sequence[float]]):
        result = np.exp(-1 * np.inner(np.abs(coordinates - self.midPoint), self.coeffs))
        self.check_vectorization(coordinates, result)
        return result

    def getAnalyticSolutionIntegral(self, start, end):
        result = 1
        for d in range(self.dim):
            one_d_integral = 0
            if start[d] < self.midPoint[d]:
                if end[d] < self.midPoint[d]:
                    one_d_integral += (np.exp(self.coeffs[d] * (end[d] - self.midPoint[d]))) / self.coeffs[d] - (
                        np.exp(self.coeffs[d] * (start[d] - self.midPoint[d]))) / self.coeffs[d]
                else:
                    one_d_integral += 1.0 / self.coeffs[d] - (
                        np.exp(self.coeffs[d] * (start[d] - self.midPoint[d]))) / self.coeffs[d]
            if end[d] > self.midPoint[d]:
                if start[d] > self.midPoint[d]:
                    one_d_integral += (np.exp(self.coeffs[d] * (self.midPoint[d] - start[d]))) / self.coeffs[d] - (
                        np.exp(self.coeffs[d] * (self.midPoint[d] - end[d]))) / self.coeffs[d]
                else:
                    one_d_integral += 1.0 / self.coeffs[d] - (
                        np.exp(self.coeffs[d] * (self.midPoint[d] - end[d]))) / self.coeffs[d]
            result *= one_d_integral
        return result


# This function is the test case function 2 of the paper from Jakeman and Roberts: https://arxiv.org/pdf/1110.0010.pdf
# It is also know as the Gaussian family of the Genz functions
class GenzGaussian(Function):
    def __init__(self, midpoint, coefficients):
        super().__init__()
        self.midpoint = midpoint
        self.coefficients = coefficients

    def eval(self, coordinates):
        dim = len(coordinates)
        assert (dim == len(self.coefficients))
        summation = 0.0
        for d in range(dim):
            summation -= self.coefficients[d] * (coordinates[d] - self.midpoint[d]) ** 2
        return np.exp(summation)

    def eval_vectorized(self, coordinates: Sequence[Sequence[float]]):
        result = np.exp(-1 * np.inner((coordinates - self.midpoint) ** 2, self.coefficients))
        self.check_vectorization(coordinates, result)
        return result

    def getAnalyticSolutionIntegral(self, start, end):
        dim = len(start)
        # print lowerBounds,upperBounds,coefficients, midpoints
        result = 1.0
        sqPiHalve = np.sqrt(np.pi) * 0.5
        for d in range(dim):
            result = result * (
                    sqPiHalve * scipy.special.erf(np.sqrt(self.coefficients[d]) * (end[d] - self.midpoint[d])) -
                    sqPiHalve * scipy.special.erf(
                np.sqrt(self.coefficients[d]) * (start[d] - self.midpoint[d]))) / np.sqrt(self.coefficients[d])
        return result


# This function is the first test function of the paper from Gerstner and Griebel:
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.33.3141&rep=rep1&type=pdf
class FunctionExpVar(Function):
    def eval(self, coordinates):
        dim = len(coordinates)
        prod = 1.0
        for d in range(dim):
            prod *= coordinates[d] ** (1.0 / dim)
        return (1 + 1.0 / dim) ** dim * prod

    def eval_vectorized(self, coordinates: Sequence[Sequence[float]]):
        dim = len(coordinates[0])
        temp = coordinates ** (1.0/dim)
        result = (1 + 1.0/dim) ** dim * np.prod(temp, axis=-1)
        self.check_vectorization(coordinates, result)
        return result

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
        super().__init__()
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
