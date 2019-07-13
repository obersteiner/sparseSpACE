from math import log2
import scipy.integrate as integrate

class BSpline(object):
    def __init__(self, p, index, knots):
        self.p = p
        self.knots = knots
        self.index = index
        assert(index <= len(knots) - p - 2)

    def __call__(self, x):
        return self.recursive_eval(x, self.p, self.index)

    def recursive_eval(self, x, p, k):
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

class LagrangeBasis(object):
    def __init__(self, p, index, knots):
        self.p = p
        self.knots = knots
        self.index = index
        #assert(index <= len(knots) - p - 2)

    def __call__(self, x):
        result = 1
        for i, knot in enumerate(self.knots):
            if self.index != i:
                result *= (x - self.knots[i]) / (self.knots[self.index] - self.knots[i])
        return result

from math import log2

class HierarchicalNotAKnotBSpline(object):
    def __init__(self, p, index, level, knots):
        self.p = p
        self.index = index
        self.level = level
        self.knots = knots
        if self.level < log2(self.p+1):
            self.spline = LagrangeBasis(p, index, knots)
        else:
            self.spline = BSpline(p, index, knots)

    def __call__(self, x):
        return self.spline(x)

    def get_integral(self, a, b):
        #if self.p == 1:
        #    if 2**self.level > self.index > 0:
        #        return 0.5**self.level * (b-a)
        #    else:
        #        return 0.5**(self.level + 1) * (b-a)
        #print(max(a,self.knots[self.index]), min(b, self.knots[self.index + (self.p+1)]), self.knots[self.index + int((self.p+1)/2)])
        #print(integrate.quad(self, max(a,self.knots[self.index]), min(b, self.knots[self.index + (self.p+1)])))
        if self.level < log2(self.p + 1):
            #print(integrate.quad(self, a, b))
            return integrate.quad(self, a, b)[0]
        else:
            #print(integrate.quad(self, max(a,self.knots[self.index]), min(b, self.knots[self.index + (self.p+1)])))
            return integrate.quad(self, max(a,self.knots[self.index]), min(b, self.knots[self.index + (self.p+1)]))[0]

class ModifiedHierarchicalNotAKnotBSpline(object):
    def __init__(self, p, index, level, knots):
        pass
    #toDo