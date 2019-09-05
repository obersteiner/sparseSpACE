import chaospy as cp
import numpy as np
import matplotlib.pyplot as plotter
import sys
import time
import os
from math import isclose

import PredatorPreyCommon as pp

# Load spatially adaptive sparse grid related files
sys.path.append('../src/')
from Function import *
from spatiallyAdaptiveSingleDimension2 import *
from ErrorCalculator import *
from GridOperation import *


# Settings
# ~ silent_mode = True
silent_mode = False

#predator = coyote
#prey = sheep

voracity = 0.00012
coyote_Px0 = 50

sigma_voracity = 0.000002
sigma_coyote_Px0 = 5

# Maximum PCE polynomial degree
poly_deg_max = pp.poly_deg_max

# Distributions information to be passed to the UncertaintyQuantification Operation
distris = [
    ("Normal", voracity, sigma_voracity),
    # ~ ("Normal", sheeps_Px0, sigma_sheeps_Px0),
    ("Normal", coyote_Px0, sigma_coyote_Px0)
]
dim = len(distris)
# Normal distribution requires infinite boundaries
a = np.array([-np.inf for _ in range(dim)])
b = np.array([np.inf for _ in range(dim)])

def get_solver_values(input_values):
    return np.concatenate(pp.get_solver_value2D(input_values))
problem_function = FunctionCustom(get_solver_values, output_dim=len(pp.time_points)*2)

# This function is later required to bring calculated values into the right shape
def reshape_result_values(vals):
    mid = int(len(vals) / 2)
    predators, preys = vals[:mid], vals[mid:]
    return np.array([predators, preys]).T

use_proxy = False
'''
if use_proxy:
    time_points_proxy = np.linspace(0, T, 31)
    # ~ time_points_proxy = time_points
    def get_solver_values_proxy(input_values):
        voracity_sample, sheep_Px0_sample, coyote_Px0_sample = input_values
        # y contains the predator solutions and prey solutions for all time values
        y = solver(voracity_sample, [coyote_Px0_sample, sheep_Px0_sample], f, time_points_proxy).y
        return np.concatenate(y)
    proxy_function = FunctionCustom(get_solver_values_proxy)
'''

# Create the Operation
op = UncertaintyQuantificationTesting(None, distris, a, b, dim=dim)

'''
# Reference solutions
problem_function_wrapped = FunctionCustom(lambda x: problem_function(x), output_dim=problem_function.output_length())
op.f = problem_function_wrapped
# ~ E_ref, Var_ref = op.calculate_expectation_and_variance_reference(mode="StandardcombiGauss")

expectations = [distr[1] for distr in distris]
standard_deviations = [distr[2] for distr in distris]
grid = GaussHermiteGrid(expectations, standard_deviations)
op.set_grid(grid)
combiinstance = StandardCombi(a, b, operation=op)
refinement_function = op.get_expectation_variance_Function()
combiinstance.perform_combi(1, 5, refinement_function)
nodes, weights = combiinstance.get_points_and_weights()
# ~ print("nodes", nodes)
# ~ print(nodes, weights)
# ~ combiinstance.print_resulting_combi_scheme(markersize=5)
# ~ combiinstance.print_resulting_sparsegrid(markersize=10)
E_ref, Var_ref = op.calculate_expectation_and_variance_for_weights(nodes.T, weights)

E_ref = reshape_result_values(E_ref)
Var_ref = reshape_result_values(Var_ref)
np.save("sparse_gauss_2D_solutions.npy", [E_ref, Var_ref])
#'''

# ~ E_pX_ref, P10_pX_ref, P90_pX_ref, Var_pX_ref = np.load("gauss_solutions.npy")
E_pX_ref, P10_pX_ref, P90_pX_ref, Var_pX_ref = np.load("gauss_2D_solutions.npy")

types = ("Gauss", "adaptiveTrapez", "adaptiveHO", "Trapez")

def run_test(testi, typid, exceed_evals=None, evals_end=None, max_time=None):
    # ~ if use_proxy:
        # ~ proxy_function_wrapped = FunctionCustom(lambda x: proxy_function(x), o)
    problem_function_wrapped = FunctionCustom(lambda x: problem_function(x), output_dim=problem_function.output_length())
    op.f = problem_function_wrapped

    multiple_evals = None
    typ = types[typid]
    if typ != "Gauss":
        if typ == "adaptiveHO":
            grid = GlobalHighOrderGridWeighted(a, b, op, boundary=False, modified_basis=False)
        elif typ == "adaptiveLagrange":
            grid = GlobalLagrangeGridWeighted(a, b, op, boundary=False)
        elif typ in ("adaptiveTrapez", "Trapez"):
            grid = GlobalTrapezoidalGridWeighted(a, b, op, boundary=False)
        op.set_grid(grid)
        combiinstance = SpatiallyAdaptiveSingleDimensions2(a, b, operation=op,
            norm=2)
        error_operator = ErrorCalculatorSingleDimVolumeGuided()
        f_refinement = op.get_PCE_Function(poly_deg_max)
        # ~ if use_proxy:
            # ~ op.f = proxy_function_wrapped
            # ~ f_refinement = op.get_expectation_variance_Function()
            # ~ op.f = -1
        # ~ else:
            # ~ f_refinement = op.get_expectation_variance_Function()

        lmax = 3
        if typ == "Trapez":
            lmax = testi + 2
        if evals_end is not None and typ != "Trapez":
            multiple_evals = dict()
            combiinstance.performSpatiallyAdaptiv(1, lmax, f_refinement,
                error_operator, tol=0, max_evaluations=evals_end,
                print_output=True, solutions_storage=multiple_evals,
                max_time=max_time)
        elif exceed_evals is None or typ == "Trapez":
            combiinstance.performSpatiallyAdaptiv(1, lmax, f_refinement,
                error_operator, tol=0,
                max_evaluations=1,
                print_output=False)
        else:
            combiinstance.performSpatiallyAdaptiv(1, lmax, f_refinement,
                error_operator, tol=np.inf,
                max_evaluations=np.inf, min_evaluations=exceed_evals+1,
                print_output=False)

        if multiple_evals is None:
            # Calculate the gPCE using the nodes and weights from the refinement
            op.f = problem_function_wrapped
            op.calculate_PCE(poly_deg_max, combiinstance)
    else:
        op.calculate_PCE_chaospy(poly_deg_max, testi+1)

    tmpdir = os.getenv("XDG_RUNTIME_DIR")
    results_path = tmpdir + "/uqtest.npy"
    solutions_data = []
    if os.path.isfile(results_path):
        solutions_data = list(np.load(results_path, allow_pickle=True))

    def calculate_errors(num_evals):
        ##extract the statistics
        # expectation value
        E_pX = reshape_result_values(op.get_expectation_PCE())
        # percentiles
        P10_pX = reshape_result_values(op.get_Percentile_PCE(10, 10*5))
        P90_pX = reshape_result_values(op.get_Percentile_PCE(90, 10*5))
        # variance
        Var = reshape_result_values(op.get_variance_PCE())

        E_predator, E_prey = E_pX.T
        P10_predator, P10_prey = P10_pX.T
        P90_predator, P90_prey = P90_pX.T
        Var_predator, Var_prey = Var.T
        def calc_error(vals, reference_vals):
            return np.array([abs(vals[i] - sol) for i,sol in enumerate(reference_vals)])
        def calc_error_relative(vals, reference_vals):
            errs = calc_error(vals, reference_vals)
            return np.array([abs(errs[i] / sol) if not isclose(sol, 0.0) else errs[i] for i,sol in enumerate(reference_vals)])
        errorr_E_predator = calc_error_relative(E_predator, E_pX_ref.T[0])
        errorr_E_prey = calc_error_relative(E_prey, E_pX_ref.T[1])
        error_P10_predator = calc_error(P10_predator, P10_pX_ref.T[0])
        error_P10_prey = calc_error(P10_prey, P10_pX_ref.T[1])
        error_P90_predator = calc_error(P90_predator, P90_pX_ref.T[0])
        error_P90_prey = calc_error(P90_prey, P90_pX_ref.T[1])
        error_Var_predator = calc_error(Var_predator, Var_pX_ref.T[0])
        error_Var_prey = calc_error(Var_prey, Var_pX_ref.T[1])

        # ~ def mean_squared_error(data):
            # ~ return np.sum([v*v for v in data]) / len(data)
        def mean_error(data):
            return np.sum(data) / len(data)
        mean_errs = (
            mean_error(errorr_E_prey), mean_error(errorr_E_predator),
            mean_error(error_P10_prey), mean_error(error_P10_predator),
            mean_error(error_P90_prey), mean_error(error_P90_predator),
            mean_error(error_Var_prey), mean_error(error_Var_predator)
        )
        if not silent_mode:
            mean_err_descs = ("E prey relative", "E predator relative", "P10 prey", "P10 predator",
                "P90 prey", "P90 predator", "Var prey", "Var predator")
            for i,desc in enumerate(mean_err_descs):
                print(f"{desc} mean error: {mean_errs[i]:.5g}")

        # ~ num_evals = proxy_function_wrapped.get_f_dict_size()
        result_data = (num_evals, typid, mean_errs)
        assert len(result_data) == 3
        return result_data

    if multiple_evals is None:
        num_evals = problem_function_wrapped.get_f_dict_size()
        result_data = calculate_errors(num_evals)

        if all([any([d[i] != result_data[i] for i in range(2)]) for d in solutions_data]):
            solutions_data.append(result_data)
            np.save(results_path, solutions_data)
        return num_evals

    solutions = op.sort_multiple_solutions(multiple_evals)
    for num_evals, integrals in solutions:
        op.calculate_PCE_from_multiple(combiinstance, integrals)
        result_data = calculate_errors(num_evals)

        if all([any([d[i] != result_data[i] for i in range(2)]) for d in solutions_data]):
            solutions_data.append(result_data)
    np.save(results_path, solutions_data)

    return problem_function_wrapped.get_f_dict_size()


evals_end = 900
max_time = 60 * 5

skip_types = ("Trapez",)

for typid in reversed(range(len(types))):
    typ = types[typid]
    print("")
    if typ in skip_types:
        print("Skipping", typ)
        continue
    print("Calculations for", typ)
    testi = 0
    start_time = time.time()
    evals_num = run_test(testi, typid, evals_end=evals_end, max_time=max_time)
    while evals_num < evals_end and time.time() - start_time < max_time:
        testi = testi+1
        print(f"last evals: {evals_num}, testi {testi}")
        evals_num = run_test(testi, typid, exceed_evals=evals_num)
