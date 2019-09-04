import chaospy as cp
import numpy as np
import scipy.integrate as ode
import matplotlib.pyplot as plotter
import sys
import time
import os
from math import isclose

# Load spatially adaptive sparse grid related files
sys.path.append('../src/')
from Function import *
from spatiallyAdaptiveSingleDimension2 import *
from ErrorCalculator import *
from GridOperation import *
from StandardCombi import *

shifted = False

error_operator = ErrorCalculatorSingleDimVolumeGuided()
if shifted:
    problem_function = FunctionUQShifted()
else:
    problem_function = FunctionUQ()
# ~ a = -np.ones(3)
# ~ b = np.ones(3)
a = np.array([-np.inf] * 3)
b = np.array([np.inf] * 3)
# ~ f = FunctionCustom(lambda p: problem_function([p[0], p[1], 0]))
# ~ f.plot([-1,-1], [1,1])
boundary = False
verbose = False
lmax = 2
adaptive_version = None

# ~ distris = [
    # ~ "Uniform", "Uniform", "Uniform"
# ~ ]
assert not shifted
normaldistr = ("Normal", 0.2, 1.0)
distris = [normaldistr for _ in range(3)]
op = UncertaintyQuantificationTesting(None, distris, a, b, dim=3)

'''
# Reference solutions
problem_function_wrapped = FunctionCustom(lambda x: problem_function(x), output_dim=problem_function.output_length())
op.f = problem_function_wrapped
# ~ E_ref, Var_ref = op.calculate_expectation_and_variance_reference(mode="StandardcombiGauss")

if False:
    grid = GaussLegendreGrid(a, b)
    op.set_grid(grid)
    combiinstance = StandardCombi(a, b, operation=op)
    combiinstance.perform_combi(1, 20, op.get_expectation_variance_Function())
    nodes, weights = combiinstance.get_points_and_weights()
    weights = weights / np.prod(b - a)
    E_ref, Var_ref = op.calculate_expectation_and_variance_for_weights(nodes.T, weights)
else:
    E_ref, Var_ref = op.calculate_expectation_and_variance_reference(modeparams=2**18)

print(E_ref, Var_ref)
if shifted:
    np.save("function_uq_shifted.npy", [E_ref, Var_ref])
else:
    np.save("function_uq.npy", [E_ref, Var_ref])
#'''

types = ("Gauss", "adaptiveTrapez", "adaptiveHO", "adaptiveLagrange", "sparseGauss", "adaptiveTransTrapez", "Halton")
typids = dict()
for i,v in enumerate(types):
    typids[v] = i

if shifted:
    E_ref, Var_ref = np.load("function_uq_shifted.npy")
else:
    E_ref, Var_ref = np.load("function_uq.npy")

def error_absolute(v, ref): return abs(ref - v)
def error_relative(v, ref): return error_absolute(v, ref) / abs(ref)

def run_test(testi, typid, exceed_evals=None, evals_end=None, max_time=None):
    problem_function_wrapped = FunctionCustom(lambda x: problem_function(x), output_dim=problem_function.output_length())
    op.f = problem_function_wrapped

    measure_start = time.time()
    multiple_evals = None
    typ = types[typid]
    if typ not in ("Gauss", "Fejer", "sparseGauss", "Halton"):
        do_inverse_transform = typ in ("adaptiveTransBSpline", "adaptiveTransTrapez", "adaptiveTransHO")
        if do_inverse_transform:
            a_trans, b_trans = np.zeros(3), np.ones(3)

        if typ == "adaptiveHO":
            grid = GlobalHighOrderGridWeighted(a, b, op, boundary=boundary)
        elif typ in ("adaptiveTrapez", "Trapez"):
            grid = GlobalTrapezoidalGridWeighted(a, b, op, boundary=boundary)
        elif typ == "adaptiveLagrange":
            grid = GlobalLagrangeGridWeighted(a, b, op, boundary=boundary)
        elif typ == "adaptiveTransBSpline":
            grid = GlobalBSplineGrid(a_trans, b_trans, boundary=boundary)
        elif typ == "adaptiveTransTrapez":
            grid = GlobalTrapezoidalGrid(a_trans, b_trans, boundary=boundary)
        elif typ == "adaptiveTransHO":
            grid = GlobalHighOrderGrid(a_trans, b_trans, boundary=boundary, split_up=False)
        op.set_grid(grid)

        f_refinement = op.get_expectation_variance_Function()
        if do_inverse_transform:
            # Use Integration operation
            f_refinement = op.get_inverse_transform_Function(f_refinement)
            # ~ f_refinement.plot(np.array([0.001]*2), np.array([0.999]*2), filename="trans.pdf")
            op_integration = Integration(f_refinement, grid, 3)
            combiinstance = SpatiallyAdaptiveSingleDimensions2(a_trans, b_trans, operation=op_integration, norm=2, version=adaptive_version)
        else:
            combiinstance = SpatiallyAdaptiveSingleDimensions2(a, b, operation=op, norm=2, version=adaptive_version)
            # ~ f_refinement = op.get_PCE_Function(poly_deg_max)

        if typ == "Trapez":
            assert False
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
        # Calculate the gPCE using the nodes and weights from the refinement
        # ~ op.calculate_PCE(None, combiinstance)
        if multiple_evals is None:
            E, Var = op.calculate_expectation_and_variance(combiinstance)
    else:
        if typ == "Gauss":
            nodes, weights = cp.generate_quadrature(testi,
                op.distributions_joint, rule="G")
        elif typ == "Halton":
            num_points = exceed_evals or 1
            num_points = int(math.ceil(num_points * 1.5))
            nodes = op.distributions_joint.sample(num_points, rule="H")
            num_samples = len(nodes[0])
            assert num_points == num_samples
            w = 1.0 / num_samples
            weights = np.array([w for _ in range(num_samples)])
        elif typ == "Fejer":
            nodes, weights = cp.generate_quadrature(testi,
                op.distributions_joint, rule="F", normalize=True)
        elif typ == "sparseGauss":
            expectations = [distr[1] for distr in distris]
            standard_deviations = [distr[2] for distr in distris]
            hgrid = GaussHermiteGrid(expectations, standard_deviations)
            op.set_grid(hgrid)
            combiinstance = StandardCombi(a, b, operation=op)
            level = testi+1
            combiinstance.perform_combi(1, level, op.get_expectation_variance_Function())
            nodes, weights = combiinstance.get_points_and_weights()
            nodes = nodes.T
            # ~ grid = GaussLegendreGrid(a, b)
            # ~ weights = weights / np.prod(b - a)
        E, Var = op.calculate_expectation_and_variance_for_weights(nodes, weights)

    print("simulation time: " + str(time.time() - measure_start) + " s")

    def calc_errors(E, Var, num_evals):
        err_descs = ("E", "Var")
        err_data = (
            (E, E_ref),
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

        result_data = (num_evals, typid, errors)
        assert len(result_data) == 3
        assert len(errors) == 4
        return result_data

    tmpdir = os.getenv("XDG_RUNTIME_DIR")
    results_path = tmpdir + "/uqtestFUQ.npy"
    solutions_data = []
    if os.path.isfile(results_path):
        solutions_data = list(np.load(results_path, allow_pickle=True))

    if multiple_evals is None:
        num_evals = problem_function_wrapped.get_f_dict_size()
        result_data = calc_errors(E, Var, num_evals)
        if all([any([d[i] != result_data[i] for i in range(2)]) for d in solutions_data]):
            solutions_data.append(result_data)
            np.save(results_path, solutions_data)

        return num_evals

    solutions = op.calculate_multiple_expectation_and_variance(multiple_evals)
    for num_evals, E, Var in solutions:
        result_data = calc_errors(E, Var, num_evals)
        if all([any([d[i] != result_data[i] for i in range(2)]) for d in solutions_data]):
            solutions_data.append(result_data)
    np.save(results_path, solutions_data)
    return problem_function_wrapped.get_f_dict_size()


evals_end = 4000
max_time = 30

# For testing
skip_types = ("adaptiveLagrange",)
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


