import numpy as np
import chaospy as cp  # Only cp.around is used here.
import math

import sys
sys.path.append('../')
from Function import *
from spatiallyAdaptiveSingleDimension2 import *
from ErrorCalculator import *
from GridOperation import *

calc_E_Var = True
# ~ calc_E_Var = False
do_PCE_func = True
# ~ do_PCE_func = False
# ~ do_HighOrder = True
do_HighOrder = False

assert calc_E_Var or do_PCE_func

# Only plot when using the ipython notebook
plot_things = 'ipykernel' in sys.modules


# This converts a number together with a description and reference value
# to a string which shows these values and errors
def get_numbers_info(description, value, reference_value=None):
	text = f"{description:s}: "
	if reference_value is None:
		if len(value) == 1:
			return text + f"{value[0]:.4g}\n"
		text += "\n"
		for i, v in enumerate(value):
			text += f"\t{v:.4g}\n"
		return text
	if len(value) == 1:
		error_abs = abs(value[0] - reference_value[0])
		text += f"{value[0]:.4g}, reference: {reference_value[0]:.4g}, abs. error: {error_abs:.4g}"
		if reference_value[0] == 0:
			return text + "\n"
		error_rel = error_abs / abs(reference_value[0])
		return text + f", rel. error: {error_rel:.4g}\n"
	text += "\n"
	for i, v in enumerate(value):
		error_abs = abs(v - reference_value[i])
		text += f"\t{v:.4g}, reference: {reference_value[i]:.4g}, abs. error: {error_abs:.4g}"
		if reference_value[i] == 0:
			text += "\n"
			continue
		error_rel = error_abs / abs(reference_value[i])
		text += f", rel. error: {error_rel:.4g}\n"
	return text


def get_combiinstance(a, b, op, boundary):
	if do_HighOrder:
		grid = GlobalHighOrderGridWeighted(a, b, op, boundary=boundary, modified_basis=False)
	else:
		grid = GlobalTrapezoidalGridWeighted(a, b, op, boundary=boundary)
	combiinstance = SpatiallyAdaptiveSingleDimensions2(a, b, operation=op,
		norm=2, grid=grid)
	return combiinstance


def calculate_reference_solutions(f, op, a, b, solutions):
	print("calculating reference solutions…")
	assert f.output_length() == 1, "f must have a single dimensional output"
	pdf_function = op.get_pdf_Function()
	expectation_func = FunctionUQWeighted(f, pdf_function)
	reference_expectation = expectation_func.getAnalyticSolutionIntegral(a, b)
	mom2_func = FunctionUQWeighted(op.get_moment_Function(2), pdf_function)
	mom2 = mom2_func.getAnalyticSolutionIntegral(a, b)
	reference_variance = mom2 - reference_expectation ** 2
	print("calculated reference E and Var:", reference_expectation, reference_variance)
	if solutions is not None:
		# Ensure that the calculated values do not differ significantly from
		# the solutions arguments
		assert solutions[0] is None or abs(reference_expectation - solutions[0]) < 10 ** -2
		assert solutions[1] is None or abs(reference_variance - solutions[1]) < 10 ** -2
	return reference_expectation, reference_variance


def plot_function(f, op, a, b, inf_borders):
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


# A helper function to reduce duplicate code
def do_test(d, a, b, f, distris, boundary=True, lmax=2, solutions=None, calculate_solutions=False):
	op = UncertaintyQuantification(f, distris, a, b, dim=d)

	reference_expectation = None
	reference_variance = None
	if calculate_solutions:
		reference_expectation, reference_variance = calculate_reference_solutions(f, op, a, b, solutions)
	elif solutions is not None:
		reference_expectation = solutions[0]
		reference_variance = solutions[1]

	inf_borders = any([math.isinf(v) for v in list(a) + list(b)])
	if plot_things:
		plot_function(f, op, a, b, inf_borders)

	can_plot = plot_things and not inf_borders

	if reference_expectation is not None:
		if not hasattr(reference_expectation, "__iter__"):
			reference_expectation = [reference_expectation]
	if reference_variance is not None:
		if not hasattr(reference_variance, "__iter__"):
			reference_variance = [reference_variance]

	error_operator = ErrorCalculatorSingleDimVolumeGuided()
	min_evals = 0
	max_evals = 340
	tol = 10**-3
	poly_deg_max = 3

	E, Var = [], []
	if calc_E_Var:
		expectation_var_func = op.get_expectation_variance_Function()
		combiinstance = get_combiinstance(a, b, op, boundary)
		print("performSpatiallyAdaptiv…")
		reference_solution = None
		if reference_expectation is not None and reference_variance is not None:
			mom2 = [reference_variance[i] + ex * ex for i, ex in enumerate(reference_expectation)]
			reference_solution = np.concatenate([reference_expectation, mom2])
		combiinstance.performSpatiallyAdaptiv(1, lmax, expectation_var_func,
			error_operator, tol=tol,
			max_evaluations=max_evals, reference_solution=reference_solution,
			min_evaluations=min_evals, do_plot=can_plot)

		print("calculate_expectation_and_variance…")
		E, Var = op.calculate_expectation_and_variance(combiinstance)
		print(f"E: {E}, Var: {Var}\n")

	print("calculate_PCE…")
	if do_PCE_func:
		combiinstance = get_combiinstance(a, b, op, boundary)
		f_pce = op.get_PCE_Function(poly_deg_max)
		combiinstance.performSpatiallyAdaptiv(1, lmax, f_pce, error_operator, tol=tol,
			max_evaluations=max_evals, reference_solution=None,
			min_evaluations=min_evals, do_plot=can_plot)
	op.calculate_PCE(poly_deg_max, combiinstance)
	print("gPCE is ", cp.around(op.get_gPCE(), 3))

	E_PCE, Var_PCE = op.get_expectation_and_variance_PCE()
	first_sens = op.get_first_order_sobol_indices()
	total_sens = op.get_total_order_sobol_indices()

	print("calculate_PCE_chaospy…")
	op.calculate_PCE_chaospy(poly_deg_max, 12)
	print("non-sparsegrid gPCE is ", cp.around(op.get_gPCE(), 3))
	E_PCE2, Var_PCE2 = op.get_expectation_and_variance_PCE()
	first_sens2 = op.get_first_order_sobol_indices()
	total_sens2 = op.get_total_order_sobol_indices()

	print("\n" +
		get_numbers_info("Expectation", E, reference_expectation) +
		get_numbers_info("Variance", Var, reference_variance) +
		get_numbers_info("PCE E", E_PCE, reference_expectation) +
		get_numbers_info("PCE Var", Var_PCE) +
		"first order sensitivity indices {}\n"
		"total order sensitivity indices {}\n".format(first_sens,
			total_sens) +
		get_numbers_info("non-sparsegrid PCE E", E_PCE2, reference_expectation) +
		get_numbers_info("non-sparsegrid PCE Var", Var_PCE2, reference_variance) +
		"non-sparsegrid first order sensitivity indices {}\n"
		"non-sparsegrid total order sensitivity indices {}\n"
		"".format(
			first_sens2,
			total_sens2))


def test_normal_integration():
	print("Calculating the expectation with an Integration Operation")
	d = 2
	bigvalue = 7.0
	a = np.array([-bigvalue, -bigvalue])
	b = np.array([bigvalue, bigvalue])

	# Whether to change weights for obtaining a higher order quadrature
	high_order = True

	distr = []
	for _ in range(d):
		distr.append(cp.Normal(0,2))
	distr_joint = cp.J(*distr)
	f = FunctionMultilinear([2.0, 0.0])
	fw = FunctionCustom(lambda coords: f(coords)[0]
		* float(distr_joint.pdf(coords)))

	# ~ grid = TrapezoidalGrid(a=a, b=b, dim=d)
	grid = GlobalBSplineGrid(a, b)
	op = Integration(fw, grid="Hund", dim=d)

	error_operator = ErrorCalculatorSingleDimVolumeGuided()
	combiinstance = SpatiallyAdaptiveSingleDimensions2(a, b, operation=op,
		do_high_order=high_order, grid=grid)
	print("performSpatiallyAdaptiv…")
	v = combiinstance.performSpatiallyAdaptiv(1, 2, fw, error_operator, tol=10**-3,
		max_evaluations=40, min_evaluations=25, do_plot=plot_things)
	integral = v[3][0]
	print("expectation", integral)


# Very simple, can be used to test what happens when the variance is zero
def test_constant():
	print("Testing a simple constant function with uniform distribution")
	d = 2
	a = np.zeros(d)
	b = np.ones(d)
	f = ConstantValue(3.0)

	do_test(d, a, b, f, "Uniform", solutions=(3.0, 0.0))


def test_constant_triangle():
	print("Testing a simple constant function with uniform distribution")
	d = 2
	a = np.zeros(d)
	b = np.ones(d)
	f = ConstantValue(3.0)

	do_test(d, a, b, f, ("Triangle", 0.75), solutions=(3.0, 0.0))


def test_linear():
	print("Testing a simple linear function with uniform distribution")
	d = 2
	a = np.zeros(d)
	b = np.ones(d)
	f = FunctionMultilinear([2.0, 0.0])

	do_test(d, a, b, f, "Uniform", solutions=(1.0, 0.3333333333333333))


def test_normal_vagebounds():
	print("Invalid corner case: normal distribution on linear function with calculated boundaries")
	d = 2
	f = FunctionMultilinear([2.0, 0.0])
	op = UncertaintyQuantification(f, ("Normal", 0, 2), np.array([-1.0, -1.0]), np.array([1.0, 1.0]), dim=d)
	a, b = op.get_boundaries(0.01)
	print("Boundaries set to", a, b)
	# The reference variance refers to 0.01 cfd threshold
	do_test(d, a, b, f, ("Normal", 0, 2), boundary=False, lmax=4, solutions=(0.0, 13.422012439469572))


def test_normal_large_border():
	print("Invalid corner case: normal distribution on linear function with large boundaries")
	d = 2
	bigvalue = 100.0
	a = np.array([-bigvalue, -bigvalue])
	b = np.array([bigvalue, bigvalue])

	f = FunctionMultilinear([2.0, 0.0])

	do_test(d, a, b, f, ("Normal", 0, 2), boundary=False, solutions=(0.0, 15.99999999996796))


def test_normal_inf_border():
	print("Corner case: normal distribution on linear function with infinite boundaries")
	d = 2
	a = np.array([-math.inf] * d)
	b = np.array([math.inf] * d)

	f = FunctionMultilinear([2.0, 0.0])

	do_test(d, a, b, f, ("Normal", 0, 2), boundary=False, lmax=4, solutions=(0.0, 16.0))


def test_something():
	print("Testing triangle distribution on GenzDiscontinious")
	d = 2
	a = np.zeros(d)
	b = np.ones(d)
	midpoint = 0.5 * np.ones(d)
	coeffs = np.array([k+1 for k in range(d)])
	f = GenzDiscontinious(border=midpoint, coeffs=coeffs)

	do_test(d, a, b, f, ("Triangle", 0.75), solutions=(0.04237441517058615, 0.01564415095611312))


def test_uq_discontinuity2D():
	print("Testing a discontinuous function")
	d = 2
	a = -np.ones(d)
	b = np.ones(d)
	f = FunctionUQ2()
	do_test(d, a, b, f, ("Triangle", 0.0), solutions=(3.2412358262581886, 10.356669220098361))


def test_uq_discontinuity3D():
	print("Testing a discontinuous function")
	d = 3
	a = -np.ones(d)
	b = np.ones(d)
	f = FunctionUQ()
	# Calculating the solutions took long here.
	do_test(d, a, b, f, ("Triangle", 0.0), solutions=(3.2412358262581886, 10.523335886765029))


def test_cantilever_beam_D():
	print("Testing a Cantilever Beam function")
	d = 3
	a = np.array([-math.inf] * d)
	b = np.array([math.inf] * d)
	f = FunctionCantileverBeamD()
	distrs = [("Normal", 2.9e7, 1.45e6), ("Normal", 500.0, 100.0), ("Normal", 1000.0, 100.0)]
	# Does not work
	do_test(d, a, b, f, distrs, boundary=False)


def test_G_function():
	print("Testing the G Function")
	d = 2
	a = np.zeros(d)
	b = np.ones(d)
	f = FunctionG(d)
	expectation = f.get_expectation()
	var = f.get_variance()
	first_order_indices = None
	# ~ first_order_indices = f.get_first_order_sobol_indices()
	print(expectation, var, first_order_indices)
	do_test(d, a, b, f, "Uniform", solutions=(expectation, var))


# ~ test_uq_discontinuity3D()
# ~ test_uq_discontinuity2D()
# ~ test_cantilever_beam_D()
# ~ test_G_function()

test_normal_integration()
# ~ test_normal_vagebounds()

# ~ test_constant_triangle()
# ~ test_constant()
# ~ test_linear()
# ~ test_something()
