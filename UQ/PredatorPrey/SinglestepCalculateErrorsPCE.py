import chaospy as cp
import numpy as np
import sys
import time
import os
from math import isclose, isinf

import PredatorPreyCommon as pp

# Load spatially adaptive sparse grid related files
sys.path.append('../../src/')
from Function import *
from spatiallyAdaptiveSingleDimension2 import *
from ErrorCalculator import *
from GridOperation import *


# Settings
timestep_problem = 25
uniform_distr = False
verbose = False
poly_deg_max = 1


voracity = 0.00012 #the voracity rate (when a predator meets sheep and kill it) (Gefraessigkeit)
sheeps_Px0 = 2000 #initial population size of sheep population
coyote_Px0 = 50 #initial population size of coyote population

# Standard deviations
sigma_voracity = 0.000002  # no uncertainty: 0.000000001, uncertainty: 0.000001
sigma_sheeps_Px0 = 1
sigma_coyote_Px0 = 5

# Distributions information to be passed to the UncertaintyQuantification Operation
# ~ dim = len(distris)
dim = 2
if uniform_distr:
    distris = [
        "Uniform", "Uniform"
    ]
    # ~ a = np.array([0.000115, 35])
    # ~ b = np.array([0.000125, 65])
    a = np.array([0, 0])
    b = np.array([1, 1])
else:
    distris = [
        ("Normal", voracity, sigma_voracity),
        # ~ ("Normal", sheeps_Px0, sigma_sheeps_Px0),
        ("Normal", coyote_Px0, sigma_coyote_Px0)
    ]
    # Normal distribution requires infinite boundaries
    a = np.array([-np.inf for _ in range(dim)])
    b = np.array([np.inf for _ in range(dim)])

'''
def get_solver_value(input_values):
    assert all([not isinf(v) for v in input_values])
    # ~ voracity_sample, sheep_Px0_sample, coyote_Px0_sample = input_values
    voracity_sample, coyote_Px0_sample = input_values
    sheep_Px0_sample = sheeps_Px0
    if voracity_sample <= 0 or coyote_Px0_sample <= 0:
        print("negative input values")
        return 0
    if uniform_distr:
        # Chaospy uniform distribution does not work with small values
        voracity_sample = 0.000115 + 0.00001 * voracity_sample
        coyote_Px0_sample = 35 + coyote_Px0_sample * 30
    # y contains the predator solutions and prey solutions for all time values
    y = solver(voracity_sample, [coyote_Px0_sample, sheep_Px0_sample], f).y
    assert len(y[0]) == len(y[1]) == len(time_points), y.shape
    return y[1][timestep_problem]
'''
def get_solver_value(input_values):
    assert not uniform_distr
    y = pp.get_solver_value2D(input_values)
    return y[1][timestep_problem]
problem_function = FunctionCustom(get_solver_value, output_dim=1)

error_operator = ErrorCalculatorSingleDimVolumeGuided()
op = UncertaintyQuantificationTesting(None, distris, a, b, dim=dim)
# ~ op = UncertaintyQuantification(problem_function, distris, a, b, dim=dim)
# ~ pa, pb = op.get_boundaries(0.01)
# ~ problem_function.plot(pa, pb, points_per_dim=5, filename="25.pdf")

types = ("Gauss", "adaptiveTrapez", "adaptiveHO", "Fejer", "adaptiveTransBSpline", "adaptiveLagrange", "sparseGauss", "adaptiveTransTrapez")
typids = dict()
for i,v in enumerate(types):
    typids[v] = i

#'''
# ~ i_ref = 256 + timestep_problem
if uniform_distr:
    E_pX_ref, P10_pX_ref, P90_pX_ref, Var_pX_ref = np.load("gauss_2D_uniform_solutions.npy")
else:
    # ~ E_pX_ref, P10_pX_ref, P90_pX_ref, Var_pX_ref = np.load("halton_2D_solutions.npy")
    E_pX_ref, P10_pX_ref, P90_pX_ref, Var_pX_ref = np.load("gauss_2D_solutions.npy")
    # ~ E_pX_ref, Var_pX_ref = np.load("sparse_gauss_2D_solutions.npy")
assert len(Var_pX_ref) == 256
assert len(Var_pX_ref[0]) == 2
E_ref = E_pX_ref[timestep_problem][1]
# ~ P10_ref = P10_pX_ref[timestep_problem][1]
# ~ P90_ref = P90_pX_ref[timestep_problem][1]
Var_ref = Var_pX_ref[timestep_problem][1]
'''
E_ref, Var_ref = np.load("step25_2D_solutions.npy")
#'''

def error_absolute(v, ref): return abs(ref - v)
def error_relative(v, ref): return error_absolute(v, ref) / abs(ref)

def run_test(testi, typid, exceed_evals=None, evals_end=None, max_time=None):
    problem_function_wrapped = FunctionCustom(lambda x: problem_function(x), output_dim=problem_function.output_length())
    op.f = problem_function_wrapped

    measure_start = time.time()
    multiple_evals = None
    typ = types[typid]
    if typ not in ("Gauss", "Fejer", "sparseGauss"):
        do_inverse_transform = typ in ("adaptiveTransBSpline", "adaptiveTransTrapez", "adaptiveTransHO")
        if do_inverse_transform:
            a_trans, b_trans = np.zeros(dim), np.ones(dim)

        if typ == "adaptiveHO":
            grid = GlobalHighOrderGridWeighted(a, b, op, boundary=uniform_distr)
        elif typ in ("adaptiveTrapez", "Trapez"):
            grid = GlobalTrapezoidalGridWeighted(a, b, op, boundary=uniform_distr)
        elif typ == "adaptiveLagrange":
            grid = GlobalLagrangeGridWeighted(a, b, op, boundary=uniform_distr)
        elif typ == "adaptiveTransBSpline":
            grid = GlobalBSplineGrid(a_trans, b_trans, boundary=uniform_distr)
        elif typ == "adaptiveTransTrapez":
            grid = GlobalTrapezoidalGrid(a_trans, b_trans, boundary=uniform_distr)
        elif typ == "adaptiveTransHO":
            grid = GlobalHighOrderGrid(a_trans, b_trans, boundary=uniform_distr, split_up=False)
        op.set_grid(grid)

        if do_inverse_transform:
            # Use Integration operation
            f_refinement = op.get_inverse_transform_Function(op.get_PCE_Function(poly_deg_max))
            # ~ f_refinement.plot(np.array([0.001]*2), np.array([0.999]*2), filename="trans.pdf")
            op_integration = Integration(f_refinement, grid, dim)
            combiinstance = SpatiallyAdaptiveSingleDimensions2(a_trans, b_trans, operation=op_integration,
                norm=2)
        else:
            combiinstance = SpatiallyAdaptiveSingleDimensions2(a, b, operation=op, norm=2)
            f_refinement = op.get_PCE_Function(poly_deg_max)

        lmax = 3
        if typ == "Trapez":
            lmax = testi + 2
        if evals_end is not None:
            multiple_evals = dict()
            combiinstance.performSpatiallyAdaptiv(1, lmax, f_refinement,
                error_operator, tol=0, max_evaluations=evals_end,
                print_output=True, solutions_storage=multiple_evals,
                max_time=max_time)
        elif exceed_evals is None or typ == "Trapez":
            combiinstance.performSpatiallyAdaptiv(1, lmax, f_refinement,
                error_operator, tol=0,
                max_evaluations=1,
                print_output=verbose)
        else:
            combiinstance.performSpatiallyAdaptiv(1, lmax, f_refinement,
                error_operator, tol=np.inf,
                max_evaluations=np.inf, min_evaluations=exceed_evals+1,
                print_output=verbose)

        # ~ combiinstance.plot()
        if multiple_evals is None:
            op.calculate_PCE(None, combiinstance)
    else:
        polys, polys_norms = cp.orth_ttr(poly_deg_max, op.distributions_joint, retall=True)
        if typ == "Gauss":
            if testi >= 29:
                # Reference solution or negative points
                return np.inf
            nodes, weights = cp.generate_quadrature(testi,
                op.distributions_joint, rule="G")
        elif typ == "Fejer":
            nodes, weights = cp.generate_quadrature(testi,
                op.distributions_joint, rule="F", normalize=True)
        elif typ == "sparseGauss":
            level = testi+1
            if level > 5:
                # normal distribution has infinite bounds
                return np.inf
            expectations = [distr[1] for distr in distris]
            standard_deviations = [distr[2] for distr in distris]
            hgrid = GaussHermiteGrid(expectations, standard_deviations)
            op.set_grid(hgrid)
            combiinstance = StandardCombi(a, b, operation=op)
            combiinstance.perform_combi(1, level, problem_function_wrapped)
            nodes, weights = combiinstance.get_points_and_weights()
            nodes = nodes.T

        f_evals = [problem_function_wrapped(c) for c in zip(*nodes)]
        op.gPCE = cp.fit_quadrature(polys, nodes, weights, np.asarray(f_evals), norms=polys_norms)

    print("simulation time: " + str(time.time() - measure_start) + " s")

    def reshape_result_values(vals): return vals[0]
    tmpdir = os.getenv("XDG_RUNTIME_DIR")
    results_path = tmpdir + "/uqtestSD.npy"
    solutions_data = []
    if os.path.isfile(results_path):
        solutions_data = list(np.load(results_path, allow_pickle=True))

    def calc_errors(op, num_evals):
        E, Var = op.get_expectation_PCE(), op.get_variance_PCE()
        E = reshape_result_values(E)
        Var = reshape_result_values(Var)

        # ~ err_descs = ("E prey", "P10 prey", "P90 prey", "Var prey")
        err_descs = ("E prey", "Var prey")
        err_data = (
            (E, E_ref),
            # ~ (P10, P10_ref),
            # ~ (P90, P90_ref),
            (Var, Var_ref)
        )
        errors = []
        for i,desc in enumerate(err_descs):
            vals = err_data[i]
            abs_err = error_absolute(*vals)
            rel_err = error_relative(*vals)
            errors.append(abs_err)
            errors.append(rel_err)
            print(f"{desc}: {vals[0]}, absolute error: {abs_err}, relative error: {rel_err}")

        result_data = (num_evals, timestep_problem, typid, errors)
        assert len(result_data) == 4
        assert len(errors) == 4
        return result_data

    if multiple_evals is None:
        num_evals = problem_function_wrapped.get_f_dict_size()
        result_data = calc_errors(op, num_evals)

        if all([any([d[i] != result_data[i] for i in range(3)]) for d in solutions_data]):
            solutions_data.append(result_data)
            np.save(results_path, solutions_data)

        return num_evals

    solutions = op.sort_multiple_solutions(multiple_evals)
    for num_evals, integrals in solutions:
        op.calculate_PCE_from_multiple(combiinstance, integrals)
        result_data = calc_errors(op, num_evals)

        if all([any([d[i] != result_data[i] for i in range(3)]) for d in solutions_data]):
            solutions_data.append(result_data)
    np.save(results_path, solutions_data)

    return problem_function_wrapped.get_f_dict_size()


evals_end = 1999
max_time = 60 * 5

# For testing
# ~ types = ("Gauss", "adaptiveTrapez", "adaptiveHO", "Fejer", "adaptiveTransBSpline", "adaptiveLagrange", "sparseGauss", "adaptiveTransTrapez")
skip_types = ("adaptiveLagrange", "Fejer", "adaptiveTransBSpline")
# ~ skip_types = ("Fejer", "adaptiveTransBSpline", "adaptiveLagrange", "sparseGauss")
assert all([typ in types for typ in skip_types])

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


