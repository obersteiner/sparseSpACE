import numpy as np
import math

import sys
sys.path.append('../')
from Function import *
from spatiallyAdaptiveSingleDimension2 import *
from ErrorCalculator import *
from GridOperation import *

# Only plot when using the ipython notebook
plot_things = 'ipykernel' in sys.modules

# A helper function to reduce duplicate code
def do_test(d, a, b, f, reference_expectation, distris, boundary=True, calc_bounds=False):
	if calc_bounds:
		op = UncertaintyQuantification(f, distris, a, b, dim=d)
		a, b = op.get_boundaries(0.01)
		print("Boundaries set to", a, b)
	op = UncertaintyQuantification(f, distris, a, b, dim=d)

	inf_borders = any([math.isinf(v) for v in list(a) + list(b)])
	if plot_things:
		pa, pb = a, b
		if inf_borders:
			# Set plot boundaries to include the place with high probability
			pa, pb = op.get_boundaries(0.01)
		print("Showing function", f)
		f.plot(pa, pb)
		print("Showing pdf")
		pdf = op.get_pdf_Function()
		pdf.plot(pa, pb, points_per_dim=11)
		print("Showing weighted function")
		weighted_f = FunctionUQWeighted(f, pdf)
		weighted_f.plot(pa, pb, points_per_dim=11)
	can_plot = plot_things and not inf_borders

	error_operator = ErrorCalculatorSingleDimVolumeGuided()
	combiinstance = SpatiallyAdaptiveSingleDimensions2(a, b, operation=op,
		boundary=boundary)
	print("performSpatiallyAdaptiv…")
	min_evals = 0
	max_evals = 30
	tol = 10**-3
	combiinstance.performSpatiallyAdaptiv(1, 2, f, error_operator, tol=tol,
		max_evaluations=max_evals, reference_solution=reference_expectation,
		min_evaluations=min_evals, do_plot=can_plot)

	print("calculate_expectation_and_variance…")
	E, Var = op.calculate_expectation_and_variance(combiinstance)
	print("E, Var", E, Var)

	poly_deg_max = 4

	print("calculate_PCE…")
	combiinstance = SpatiallyAdaptiveSingleDimensions2(a, b, operation=op,
		boundary=boundary)
	f_pce = op.get_PCE_Function(poly_deg_max)
	combiinstance.performSpatiallyAdaptiv(1, 2, f_pce, error_operator, tol=tol,
		max_evaluations=max_evals, reference_solution=reference_expectation,
		min_evaluations=min_evals, do_plot=can_plot)
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


def test_normal_integration():
	print("Calculating the expectation with an Integration Operation")
	d = 2
	bigvalue = 7.0
	# a and b are actually unused
	a = np.array([-bigvalue, -bigvalue])
	b = np.array([bigvalue, bigvalue])

	distr = []
	for _ in range(d):
		distr.append(cp.Normal(0,2))
	distr_joint = cp.J(*distr)
	f = FunctionLinearSum([2.0, 0.0])
	fw = FunctionCustom(lambda coords: f(coords)[0]
		* float(distr_joint.pdf(coords)))

	grid = TrapezoidalGrid(a=a, b=b, dim=d)
	op = Integration(fw, grid=grid, dim=d)

	error_operator = ErrorCalculatorSingleDimVolumeGuided()
	combiinstance = SpatiallyAdaptiveSingleDimensions2(a, b, operation=op)
	print("performSpatiallyAdaptiv…")
	v = combiinstance.performSpatiallyAdaptiv(1, 2, fw, error_operator, tol=10**-3,
		max_evaluations=40,
		min_evaluations=25, do_plot=plot_things)
	integral = v[3][0]
	print("expectation", integral)


# Very simple, can be used to test what happens when the variance is zero
def test_constant():
	print("Testing a simple constant function with uniform distribution")
	d = 2
	a = np.zeros(d)
	b = np.ones(d)
	f = ConstantValue(3.0)
	reference_expectation = 3.0

	do_test(d, a, b, f, reference_expectation, "Uniform")
	# ~ do_test(d, a, b, f, reference_expectation, ("Triangle", 0.75))


def test_linear():
	print("Testing a simple linear function with uniform distribution")
	d = 2
	a = np.zeros(d)
	b = np.ones(d)
	f = FunctionLinearSum([2.0, 0.0])
	reference_expectation = 1.0

	do_test(d, a, b, f, reference_expectation, "Uniform")


def test_normal():
	print("Testing normal distribution on linear function with calculated boundaries")
	d = 2
	bigvalue = 1.0
	# a and b are actually unused
	a = np.array([-bigvalue, -bigvalue])
	b = np.array([bigvalue, bigvalue])

	f = FunctionLinearSum([2.0, 0.0])
	reference_expectation = 0.000000000000000001

	do_test(d, a, b, f, reference_expectation, ("Normal", 0, 2), False, True)


def test_normal_large_border():
	print("Testing normal distribution on linear function with large boundaries")
	d = 2
	bigvalue = 100.0
	a = np.array([-bigvalue, -bigvalue])
	b = np.array([bigvalue, bigvalue])

	f = FunctionLinearSum([2.0, 0.0])
	reference_expectation = 0.0

	do_test(d, a, b, f, reference_expectation, ("Normal", 0, 2), False, False)


def test_normal_inf_border():
	print("Testing normal distribution on linear function with infinite boundaries")
	d = 2
	a = np.array([-math.inf, -math.inf])
	b = np.array([math.inf, math.inf])

	f = FunctionLinearSum([2.0, 0.0])
	reference_expectation = 0.0

	do_test(d, a, b, f, reference_expectation, ("Normal", 0, 2), False, False)


def test_something():
	print("Testing triangle distribution on GenzDiscontinious")
	d = 2
	a = np.zeros(d)
	b = np.ones(d)
	midpoint = 0.5 * np.ones(d)
	coeffs = np.array([k+1 for k in range(d)])
	f = GenzDiscontinious(border=midpoint, coeffs=coeffs)
	reference_expectation = None

	# Tests have shown: Expectation: 0.0403, Variance: 0.01559
	do_test(d, a, b, f, reference_expectation, ("Triangle", 0.75))


# ~ test_normal_integration()

test_normal_inf_border()
# ~ test_normal_large_border()
# ~ test_normal()
# ~ test_linear()
# ~ test_something()
