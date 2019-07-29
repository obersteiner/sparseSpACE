import numpy as np
import chaospy as cp
import os

import sys
sys.path.append('../')
from Function import *
from spatiallyAdaptiveSingleDimension2 import *
from ErrorCalculator import *
from GridOperation import *


def run_test(evals_num, use_spatially_adaptive, do_HighOrder):
	lmax = 2
	d = 2
	a = np.zeros(d)
	b = np.ones(d)
	f = FunctionG(d)
	reference_expectation = f.get_expectation()
	reference_variance = f.get_variance()

	op = UncertaintyQuantification(f, "Uniform", a, b, dim=d)

	if use_spatially_adaptive:
		if do_HighOrder:
			grid = GlobalHighOrderGridWeighted(a, b, op, boundary=True, modified_basis=False)
		else:
			grid = GlobalTrapezoidalGridWeighted(a, b, op, boundary=True)
		combiinstance = SpatiallyAdaptiveSingleDimensions2(a, b, operation=op,
			norm=2, grid=grid)

		error_operator = ErrorCalculatorSingleDimVolumeGuided()
		expectation_var_func = op.get_expectation_variance_Function()
		mom2 = reference_variance + reference_expectation * reference_expectation
		reference_solution = np.array([reference_expectation, mom2])
		combiinstance.performSpatiallyAdaptiv(1, lmax, expectation_var_func,
			error_operator, tol=0,
			max_evaluations=evals_num, reference_solution=reference_solution,
			print_output=1)

		(E,), (Var,) = op.calculate_expectation_and_variance(combiinstance)
	else:
		joint_distr = op.distributions_joint
		nodes, weights = cp.generate_quadrature(
			evals_num, joint_distr, sparse=True, growth=True)
		evals = [f(x)[0] for x in nodes.T]
		E = sum([v * weights[i] for i,v in enumerate(evals)])
		mom2 = sum([v * v * weights[i] for i,v in enumerate(evals)])
		Var = mom2 - E * E

	# ~ print(f"E: {E}, Var: {Var}\n")
	# ~ print("reference E and Var: ", reference_expectation, reference_variance)

	err_E = abs((E - reference_expectation) / reference_expectation)
	err_Var = abs((Var - reference_variance) / reference_variance)
	num_evals = f.get_f_dict_size()

	print("evals, relative errors:", num_evals, err_E, err_Var)

	result_data = (num_evals, use_spatially_adaptive, do_HighOrder, err_E, err_Var)

	tmpdir = os.getenv("XDG_RUNTIME_DIR")
	results_path = tmpdir + "/uqtestG.npy"
	solutions_data = []
	if os.path.isfile(results_path):
		solutions_data = list(np.load(results_path, allow_pickle=True))
	if all([any([d[i] != result_data[i] for i in range(3)]) for d in solutions_data]):
		solutions_data.append(result_data)
		np.save(results_path, solutions_data)


print("Calculating adaptive sparse grid errors for small max evals")
for i in range(1, 10):
	evals_num = 19 + i ** 2
	print("max evals: ", evals_num)
	run_test(evals_num, True, False)
	run_test(evals_num, True, True)

print("Calculating adaptive sparse grid errors for big max evals")
for i in range(6, 10):
	evals_num = 2 ** i
	print("max evals: ", evals_num)
	run_test(evals_num, True, False)
	run_test(evals_num, True, True)
# Number of points for order 8 CC
run_test(1280, True, False)
run_test(1280, True, True)

print("Calculating Clenshaw-Curtis sparse grid errors")
for i in range(1, 9):
	print("CC order: ", i)
	run_test(i, False, None)

