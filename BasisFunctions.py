from math import log2
import scipy.integrate as integrate
import numpy as np

class BSpline(object):
    def __init__(self, p, index, knots):
        self.p = p
        self.knots = knots
        self.index = index
        assert(index <= len(knots) - p - 2)

    def __call__(self, x):
        return self.recursive_eval(x, self.p, self.index)

    def recursive_eval(self, x, p, k):
        if x < self.knots[k] or x > self.knots[k+p+1]:
            return 0.0
        if p == 0:
            return self.chi(x, k)
        else:
            result = (x - self.knots[k]) / (self.knots[k + p] - self.knots[k]) * self.recursive_eval(x, p-1, k)
            result += (self.knots[k + p + 1] - x) / (self.knots[k + p + 1] - self.knots[k + 1]) * self.recursive_eval(x, p-1, k + 1)
            return result

    def chi(self, x, k):
        if self.knots[k] <= x < self.knots[k+1]:
            return 1.0
        else:
            return 0.0

    def get_first_derivative(self,x):
        return self.get_first_derivative_recursive(x, self.p, self.index)

    def get_second_derivative(self, x):
        return self.get_second_derivative_recursive(x, self.p, self.index)

    def get_first_derivative_recursive(self, x, p, k):
        if p == 0:
            return 0.0
        dh1 = 1 / (self.knots[k + p] - self.knots[k])
        dh2 = 1 / (self.knots[k + p + 1] - self.knots[k + 1])
        result = dh1 * self.recursive_eval(x, p-1, k) - dh2 * self.recursive_eval(x, p-1, k+1)
        result += (x - self.knots[k]) / (self.knots[k + p] - self.knots[k]) * self.get_first_derivative_recursive(x, p-1, k)
        result += (self.knots[k + p + 1] - x) / (self.knots[k + p + 1] - self.knots[k + 1]) * self.get_first_derivative_recursive(x, p-1, k+1)
        return result

    def get_second_derivative_recursive(self, x, p, k):
        if p <= 1:
            return 0.0
        dh1 = 1 / (self.knots[k + p] - self.knots[k])
        dh2 = 1 / (self.knots[k + p + 1] - self.knots[k + 1])
        result = 2*(dh1 * self.get_first_derivative_recursive(x, p - 1, k) - dh2 * self.get_first_derivative_recursive(x, p - 1, k + 1))
        result += (x - self.knots[k]) / (self.knots[k + p] - self.knots[k]) * self.get_second_derivative_recursive(x, p-1, k)
        result += (self.knots[k + p + 1] - x) / (self.knots[k + p + 1] - self.knots[k + 1]) * self.get_second_derivative_recursive(x, p-1, k+1)
        return result

class LagrangeBasis(object):
    def __init__(self, p, index, knots):
        self.p = p
        self.knots = knots
        self.index = index
        self.factor = 1
        for i, knot in enumerate(self.knots):
            if self.index != i:
                self.factor *= 1 / (self.knots[self.index] - self.knots[i])
        #assert(index <= len(knots) - p - 2)

    def __call__(self, x):
        result = 1
        for i, knot in enumerate(self.knots):
            if self.index != i:
                result *= (x - self.knots[i])
        return result * self.factor

    def get_first_derivative(self, x):
        return self.derivative_for_index(x, [self.index])

    def derivative_for_index(self, x, indexSet):
        result = 0.0
        for i, knot in enumerate(self.knots):
            if i not in indexSet:
                summation_factor = 1.0
                for j, knot in enumerate(self.knots):
                    if j not in indexSet and i != j:
                        summation_factor *= (x - self.knots[j]) / (self.knots[self.index] - self.knots[j])
                summation_factor *= 1/(self.knots[self.index] - self.knots[i])
                result += summation_factor
        return result

    def get_second_derivative(self, x):
        result = 0.0
        for i, knot in enumerate(self.knots):
            if self.index != i:
                result += 1/(self.knots[self.index] - self.knots[i]) * self.derivative_for_index(x, [self.index, i])
        return result

    def get_integral(self, a, b, coordsD, weightsD):
        result = 0.0

        left_border = a
        right_border = b
        coords = np.array(coordsD)
        coords += np.ones(int((self.p+1)/2))
        coords *= (right_border - left_border) / 2.0
        coords += left_border
        weights = np.array(weightsD) * (right_border - left_border) / 2
        f_evals = np.array([self(coord) for coord in coords])
        #print(coordsD, a, b, weights)
        result += np.inner(f_evals, weights)
        return result

class LagrangeBasisRestricted(LagrangeBasis):
    def __call__(self, x):
        if self.knots[max(0,self.index - 1)] <= x <= self.knots[min(self.index + 1, len(self.knots) - 1)]:
            return super().__call__(x)
        else:
            return 0.0

    def get_integral(self, a, b, coordsD, weightsD):
        result = 0.0

        left_border = self.knots[max(0,self.index - 1)]
        right_border = self.knots[min(self.index + 1, len(self.knots) - 1)]
        coords = np.array(coordsD)
        coords += np.ones(int((self.p+1)/2))
        coords *= (right_border - left_border) / 2.0
        coords += left_border
        weights = np.array(weightsD) * (right_border - left_border) / 2
        f_evals = np.array([self(coord) for coord in coords])
        #print(coordsD, a, b, weights)
        result += np.inner(f_evals, weights)
        return result

from math import log2
import numpy.polynomial.legendre as legendre

class HierarchicalNotAKnotBSpline(object):
    def __init__(self, p, index, level, knots):
        self.p = p
        self.index = index
        self.level = level
        self.knots = knots
        if self.level < log2(self.p+1):
            self.spline = LagrangeBasis(p, index, knots)
            self.startIndex = 0
            self.endIndex = len(knots) - 1
        else:
            self.spline = BSpline(p, index, knots)
            self.startIndex = index
            self.endIndex = index + p + 1

    def __call__(self, x):
        return self.spline(x)

    def get_integral(self, a, b, coordsD, weightsD):
        result = 0.0
        for i in range(self.startIndex, self.endIndex):
            if self.knots[i+1] >= a and self.knots[i] <= b:
                left_border = max(self.knots[i], a)
                right_border = min(self.knots[i+1], b)
                coords = np.array(coordsD)
                coords += np.ones(int((self.p+1)/2))
                coords *= (right_border - left_border) / 2.0
                coords += left_border
                weights = np.array(weightsD) * (right_border - left_border) / 2
                f_evals = np.array([self(coord) for coord in coords])
                #print(coordsD, a, b, weights)
                result += np.inner(f_evals, weights)

        #if self.level < log2(self.p + 1):
        #    #print(integrate.quad(self, a, b, epsabs=1.49e-20, epsrel=1.49e-14))
        #    result2 = integrate.quad(self, a, b, epsabs=1.49e-20, epsrel=1.49e-14)[0]
        #else:
        #    #print(integrate.quad(self, max(a,self.knots[self.index]), min(b, self.knots[self.index + (self.p+1)]), epsabs=1.49e-20, epsrel=1.49e-14))
        #    result2 = integrate.quad(self, max(a,self.knots[self.index]), min(b, self.knots[self.index + (self.p+1)]), epsabs=1.49e-20, epsrel=1.49e-14)[0]
        #print(result, result2)

        return result

class HierarchicalNotAKnotBSplineModified(object):
    def __init__(self, p, index, level, knots, a, b):
        self.p = p
        self.index = index
        self.level = level
        self.knots = knots
        self.a = a
        self.b = b
        if self.level < log2(self.p+1):
            self.spline = LagrangeBasis(p, index, knots)
            if self.index == 1 or self.index == 2**self.level - 1:
                self.spline2 = LagrangeBasis(self.p, 0, self.knots)
                self.spline3 = LagrangeBasis(self.p, 2**self.level, self.knots)
            self.startIndex = 0
            self.endIndex = len(knots) - 1
        else:
            self.spline = BSpline(p, index, knots)
            if self.index == 1 or self.index == 2**self.level - 1:
                self.spline2 = BSpline(self.p, 0, self.knots)
                self.spline3 = BSpline(self.p, 2**self.level, self.knots)
            self.startIndex = index
            self.endIndex = index + p + 1

    def __call__(self, x):
        if self.level == 1:
            assert(self.index == 1)
            return 1.0
        elif self.level >= 2 and (self.index == 1 or self.index == 2**self.level - 1):
            result = self.spline(x)
            if self.index == 1:
                #print(self.spline.get_second_derivative(self.a), self.spline2.get_second_derivative(self.a))
                if self.p > 1:
                    result -=self.spline.get_second_derivative(self.a)/ self.spline2.get_second_derivative(self.a) * self.spline2(x)
                else:
                    result += 2 * self.spline2(x)
            else:
                if self.p > 1:
                    result -= self.spline(x) - self.spline.get_second_derivative(self.b)/ self.spline3.get_second_derivative(self.b) * self.spline3(x)
                else:
                    result += 2 *  self.spline3(x)
            return result
        else:
            return self.spline(x)

    def get_integral(self, a, b, coordsD, weightsD):
        result = 0.0
        for i in range(self.startIndex, self.endIndex):
            if self.knots[i+1] >= a and self.knots[i] <= b:
                left_border = max(self.knots[i], a)
                right_border = min(self.knots[i+1], b)
                coords = np.array(coordsD)
                coords += np.ones(int((self.p+1)/2))
                coords *= (right_border - left_border) / 2.0
                coords += left_border
                weights = np.array(weightsD) * (right_border - left_border) / 2
                f_evals = np.array([self(coord) for coord in coords])
                #print(coordsD, a, b, weights)
                result += np.inner(f_evals, weights)

        #if self.level < log2(self.p + 1):
        #    #print(integrate.quad(self, a, b, epsabs=1.49e-20, epsrel=1.49e-14))
        #    result2 = integrate.quad(self, a, b, epsabs=1.49e-20, epsrel=1.49e-14)[0]
        #else:
        #    #print(integrate.quad(self, max(a,self.knots[self.index]), min(b, self.knots[self.index + (self.p+1)]), epsabs=1.49e-20, epsrel=1.49e-14))
        #    result2 = integrate.quad(self, max(a,self.knots[self.index]), min(b, self.knots[self.index + (self.p+1)]), epsabs=1.49e-20, epsrel=1.49e-14)[0]
        #print(result, result2)

        return result


class ModifiedHierarchicalNotAKnotBSpline(object):
    def __init__(self, p, index, level, knots):
        pass
    #toDo