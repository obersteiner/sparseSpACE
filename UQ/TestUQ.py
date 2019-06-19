import numpy as np
import math

from sys import path
path.append('../')
from Function import *
from spatiallyAdaptiveSingleDimension2 import *
from ErrorCalculator import *
from GridOperation import *


# A helper function to reduce duplicate code
def do_test(d, a, b, f, reference_expectation, distris, boundary=True):
	error_operator = ErrorCalculatorSingleDimVolumeGuided()
	grid = TrapezoidalGrid(a=a, b=b, dim=d)
	op = UncertaintyQuantification(f, distris, grid=grid, dim=d)

	combiinstance = SpatiallyAdaptiveSingleDimensions2(a, b, operation=op,
		boundary=boundary)
	print("performSpatiallyAdaptiv…")
	combiinstance.performSpatiallyAdaptiv(1, 2, f, error_operator, tol=10**-3,
		max_evaluations=40, reference_solution=reference_expectation)

	print("calculateExpectationAndVariance…")
	E, Var = op.calculateExpectationAndVariance(combiinstance)

	poly_deg_max = 4

	print("calculatePCE…")
	op.calculatePCE(poly_deg_max, combiinstance)
	E_PCE, Var_PCE = op.getExpectationAndVariancePCE()
	first_sens = op.getFirstOrderSobolIndices()
	total_sens = op.getTotalOrderSobolIndices()

	print("calculatePCE_Chaospy…")
	op.calculatePCE_Chaospy(poly_deg_max, 12)
	E_PCE2, Var_PCE2 = op.getExpectationAndVariancePCE()
	first_sens2 = op.getFirstOrderSobolIndices()
	total_sens2 = op.getTotalOrderSobolIndices()

	print("\n"
		"Expectation: {:.4g}, Variance: {:.4g}\n"
		"PCE E: {:.4g}, Var: {:.4g}\n"
		"first order sensitivity indices {}\n"
		"total order sensitivity indices {}\n"
		"non-sparsegrid PCE E: {:.4g}, Var: {:.4g}\n"
		"non-sparsegrid first order sensitivity indices {}\n"
		"non-sparsegrid total order sensitivity indices {}\n"
		"".format(
		E, Var,
		E_PCE, Var_PCE,
		first_sens,
		total_sens,
		E_PCE2, Var_PCE2,
		first_sens2,
		total_sens2))


'''
def test_triangle():
	# ~ a = np.array([])
	dim = 3
	a = [-2.0, 0.5, 0.0]
	b = [2.0, 3.0, 2.0]
	f = FunctionLinear([0.5, 3.0, -5.0])

# ~ reference_solution = f.getAnalyticSolutionIntegral(a, b)
	error_operator = ErrorCalculatorSingleDimVolumeGuided()
	grid = TrapezoidalGrid(a=a, b=b, dim=dim)
	op = UncertaintyQuantification(f, ("Triangle", 1.0), grid=grid, dim=dim)

	combiinstance = SpatiallyAdaptiveSingleDimensions2(a, b, operation=op)

	E, Var = op.calculateExpectationAndVariance(combiinstance, error_operator, 40)
# ~ E2, Var2 = op.calculateExpectationAndVariance2(combiinstance, 2, 2, f, error_operator, 10**-2)
	op.calculatePCE(4, combiinstance, error_operator, 40)
	E_PCE = op.getExpectationPCE()
	Std_PCE = op.getStdPCE()
	# ~ first_sens = op.getFirstOrderSobolIndices()
	print("expectation and variance\t", E, Var)
	# ~ print("expectation and variance 2\t", E2, Var2)
	print("PCE expectation and variance\t", E_PCE, Std_PCE ** 2)
	# ~ print("first order sensitivity indices\t", first_sens)
'''


# Very simple, can be used to test what happens when the variance is zero
def test_constant():
	d = 2
	a = np.zeros(d)
	b = np.ones(d)
	f = ConstantValue(3.0)
	reference_expectation = 3.0

	do_test(d, a, b, f, reference_expectation, "Uniform")
	# ~ do_test(d, a, b, f, reference_expectation, ("Triangle", 0.75))


def test_linear():
	d = 2
	a = np.zeros(d)
	b = np.ones(d)
	f = FunctionLinearSum([2.0, 0.0])
	reference_expectation = 1.0

	do_test(d, a, b, f, reference_expectation, "Uniform")


def test_normal():
	d = 2
	a = [-math.inf, -math.inf]
	b = [math.inf, math.inf]
	f = FunctionLinearSum([2.0, 0.0])
	reference_expectation = 1.0

	do_test(d, a, b, f, reference_expectation, ("Normal", 0, 2), False)


def test_something():
	d = 2
	a = np.zeros(d)
	b = np.ones(d)
	midpoint = 0.5 * np.ones(d)
	coeffs = np.array([k+1 for k in range(d)])
	f = GenzDiscontinious(border=midpoint, coeffs=coeffs)
	reference_expectation = None

	# Tests have shown: Expectation: 0.0403, Variance: 0.01559
	do_test(d, a, b, f, reference_expectation, ("Triangle", 0.75))


# ~ test_linear()
test_normal()
# ~ test_something()
