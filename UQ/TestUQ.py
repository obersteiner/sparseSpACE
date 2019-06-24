import numpy as np
import math

from sys import path
path.append('../')
from Function import *
from spatiallyAdaptiveSingleDimension2 import *
from ErrorCalculator import *
from GridOperation import *

plot_things = False
# ~ plot_things = True

# A helper function to reduce duplicate code
def do_test(d, a, b, f, reference_expectation, distris, boundary=True, calc_bounds=False):
	if calc_bounds:
		grid = TrapezoidalGrid(a=a, b=b, dim=d)
		op = UncertaintyQuantification(f, distris, grid=grid, dim=d)
		a, b = op.get_boundaries(0.01)
		print("Boundaries set to", a, b)
	grid = TrapezoidalGrid(a=a, b=b, dim=d)
	op = UncertaintyQuantification(f, distris, grid=grid, dim=d)

	if plot_things:
		print("Showing function", f)
		f.plot(a, b)
		print("Showing pdf")
		pdf = op.get_pdf_Function()
		pdf.plot(a, b, points_per_dim=10)
		print("Showing weighted function")
		weighted_f = FunctionUQWeighted(f, pdf)
		weighted_f.plot(a, b, points_per_dim=10)

	error_operator = ErrorCalculatorSingleDimVolumeGuided()
	combiinstance = SpatiallyAdaptiveSingleDimensions2(a, b, operation=op,
		boundary=boundary)
	print("performSpatiallyAdaptiv…")
	combiinstance.performSpatiallyAdaptiv(1, 2, f, error_operator, tol=10**-3,
		max_evaluations=40, reference_solution=reference_expectation,
		min_evaluations=25, do_plot=plot_things)

	print("calculate_expectation_and_variance…")
	E, Var = op.calculate_expectation_and_variance(combiinstance)

	poly_deg_max = 4

	print("calculate_PCE…")
	op.calculate_PCE(poly_deg_max, combiinstance)
	E_PCE, Var_PCE = op.get_expectation_and_variance_PCE()
	first_sens = op.get_first_order_sobol_indices()
	total_sens = op.get_total_order_sobol_indices()

	print("calculate_PCE_chaospy…")
	op.calculate_PCE_chaospy(poly_deg_max, 12)
	E_PCE2, Var_PCE2 = op.get_expectation_and_variance_PCE()
	first_sens2 = op.get_first_order_sobol_indices()
	total_sens2 = op.get_total_order_sobol_indices()

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
	bigvalue = 1.0
	# a and b are actually unused
	a = np.array([-bigvalue, -bigvalue])
	b = np.array([bigvalue, bigvalue])

	f = FunctionLinearSum([2.0, 0.0])
	reference_expectation = 1.0

	do_test(d, a, b, f, reference_expectation, ("Normal", 0, 2), False, True)


def test_normal_large_border():
	d = 2
	bigvalue = 10.0 ** 10
	a = np.array([-bigvalue, -bigvalue])
	b = np.array([bigvalue, bigvalue])

	f = FunctionLinearSum([2.0, 0.0])
	reference_expectation = 1.0

	do_test(d, a, b, f, reference_expectation, ("Normal", 0, 2), False, False)


def test_normal_inf_border():
	d = 2
	a = np.array([-math.inf, -math.inf])
	b = np.array([math.inf, math.inf])

	f = FunctionLinearSum([2.0, 0.0])
	reference_expectation = 1.0

	do_test(d, a, b, f, reference_expectation, ("Normal", 0, 2), False, False)


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


# ~ test_normal_inf_border()
# ~ test_normal_large_border()
# ~ test_normal()
test_linear()
# ~ test_something()
