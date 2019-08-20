import numpy as np
import chaospy as cp
import os

import sys
sys.path.append('../src/')
from Function import *
from spatiallyAdaptiveSingleDimension2 import *
from ErrorCalculator import *
from GridOperation import *


types = ("Gauss", "adaptiveTrapez", "adaptiveHO", "BSpline", "adaptiveLagrange")

d = 2
a = np.zeros(d)
b = np.ones(d)
f_g = FunctionG(d)
reference_expectation = f_g.get_expectation()
reference_variance = f_g.get_variance()
# Create the operation only once
op = UncertaintyQuantification(None, "Uniform", a, b, dim=d)

def run_test(evals_num, typid, exceed_evals=None):
	lmax = 2
	f = FunctionCustom(lambda x: f_g(x))
	op.f = f

	typ = types[typid]
	if typ != "Gauss":
		if typ == "adaptiveHO":
			grid = GlobalHighOrderGridWeighted(a, b, op, boundary=True)
		elif typ == "adaptiveTrapez":
			grid = GlobalTrapezoidalGridWeighted(a, b, op, boundary=True)
		elif typ == "adaptiveLagrange":
			grid = GlobalLagrangeGridWeighted(a, b, op, boundary=True)
		elif typ == "BSpline":
			# ~ grid = GlobalBSplineGrid(a, b, modified_basis=True, boundary=False, p=3)
			grid = GlobalBSplineGrid(a, b)
			# ~ lmax = 3
		op.set_grid(grid)
		combiinstance = SpatiallyAdaptiveSingleDimensions2(a, b, operation=op, norm=2)

		error_operator = ErrorCalculatorSingleDimVolumeGuided()
		expectation_var_func = op.get_expectation_variance_Function()
		mom2 = reference_variance + reference_expectation * reference_expectation
		reference_solution = np.array([reference_expectation, mom2])
		op.set_reference_solution(reference_solution)
		if exceed_evals is None:
			combiinstance.performSpatiallyAdaptiv(1, lmax, expectation_var_func,
				error_operator, tol=0,
				max_evaluations=evals_num,
				print_output=False)
		else:
			combiinstance.performSpatiallyAdaptiv(1, lmax, expectation_var_func,
				error_operator, tol=np.inf,
				max_evaluations=np.inf, min_evaluations=exceed_evals+1,
				print_output=False)

		(E,), (Var,) = op.calculate_expectation_and_variance(combiinstance)
	else:
		joint_distr = op.distributions_joint
		nodes, weights = cp.generate_quadrature(
			evals_num, joint_distr, rule="G")
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

	result_data = (num_evals, typid, err_E, err_Var)

	tmpdir = os.getenv("XDG_RUNTIME_DIR")
	results_path = tmpdir + "/uqtestG.npy"
	solutions_data = []
	if os.path.isfile(results_path):
		solutions_data = list(np.load(results_path, allow_pickle=True))
	if all([any([d[i] != result_data[i] for i in range(2)]) for d in solutions_data]):
		solutions_data.append(result_data)
		np.save(results_path, solutions_data)

	return num_evals


for i,typ in enumerate(types[1:]):
	typid = i+1
	print("Calculations for", typ)
	evals_num = run_test(1, typid)
	while evals_num < 1280:
		print("last evals:", evals_num)
		evals_num = run_test(None, typid, exceed_evals=evals_num)

print("Calculating convent. errors")
for i in range(1, 36):
	print("order: ", i)
	# Gauss
	run_test(i, 0)

