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

error_operator = ErrorCalculatorSingleDimVolumeGuided()
problem_function = FunctionUQ()
a = np.zeros(3)
b = np.ones(3)
# ~ f = FunctionCustom(lambda p: problem_function([p[0], p[1], 0]))
# ~ f.plot([-1,-1], [1,1])
boundary = True
verbose = False
lmax = 2

distris = [
    "Uniform", "Uniform", "Uniform"
]
op = UncertaintyQuantification(None, distris, a, b, dim=3)

'''
# Reference solutions
problem_function_wrapped = FunctionCustom(lambda x: problem_function(x), output_dim=problem_function.output_length())
op.f = problem_function_wrapped
E_ref, Var_ref = op.calculate_expectation_and_variance_reference()
print(E_ref, Var_ref)
np.save("function_uq.npy", [E_ref, Var_ref])
#'''

types = ("Gauss", "adaptiveTrapez", "adaptiveHO", "adaptiveLagrange")
typids = dict()
for i,v in enumerate(types):
    typids[v] = i

E_ref, Var_ref = np.load("function_uq.npy")

def error_absolute(v, ref): return abs(ref - v)
def error_relative(v, ref): return error_absolute(v, ref) / abs(ref)

def run_test(testi, typid, exceed_evals=None):
    problem_function_wrapped = FunctionCustom(lambda x: problem_function(x), output_dim=problem_function.output_length())
    op.f = problem_function_wrapped

    measure_start = time.time()
    typ = types[typid]
    if typ not in ("Gauss", "Fejer"):
        do_inverse_transform = typ in ("adaptiveTransBSpline", "adaptiveTransTrapez", "adaptiveTransHO")
        if do_inverse_transform:
            a_trans, b_trans = np.zeros(dim), np.ones(dim)

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

        if do_inverse_transform:
            assert False
            # Use Integration operation
            f_refinement = op.get_inverse_transform_Function(op.get_PCE_Function(poly_deg_max))
            # ~ f_refinement.plot(np.array([0.001]*2), np.array([0.999]*2), filename="trans.pdf")
            op_integration = Integration(f_refinement, grid, dim)
            combiinstance = SpatiallyAdaptiveSingleDimensions2(a_trans, b_trans, operation=op_integration,
                norm=2, grid=grid)
        else:
            combiinstance = SpatiallyAdaptiveSingleDimensions2(a, b, operation=op,
                norm=2, grid=grid)
            # ~ f_refinement = op.get_PCE_Function(poly_deg_max)
        f_refinement = op.get_expectation_variance_Function()

        if typ == "Trapez":
            assert False
        if exceed_evals is None or typ == "Trapez":
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
        E, Var = op.calculate_expectation_and_variance(combiinstance)
    else:
        if typ == "Gauss":
            nodes, weights = cp.generate_quadrature(testi,
                op.distributions_joint, rule="G")
        elif typ == "Fejer":
            nodes, weights = cp.generate_quadrature(testi,
                op.distributions_joint, rule="F", normalize=True)
        E, Var = op.calculate_expectation_and_variance_for_weights(nodes, weights)

    print("simulation time: " + str(time.time() - measure_start) + " s")

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

    num_evals = problem_function_wrapped.get_f_dict_size()
    result_data = (num_evals, typid, errors)
    assert len(result_data) == 3
    assert len(errors) == 4

    tmpdir = os.getenv("XDG_RUNTIME_DIR")
    results_path = tmpdir + "/uqtestFUQ.npy"
    solutions_data = []
    if os.path.isfile(results_path):
        solutions_data = list(np.load(results_path, allow_pickle=True))
    if all([any([d[i] != result_data[i] for i in range(2)]) for d in solutions_data]):
        solutions_data.append(result_data)
        np.save(results_path, solutions_data)

    return num_evals


# ~ evals_end = 900
evals_end = 1200

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
    evals_num = run_test(testi, typid)
    while evals_num < evals_end:
        testi = testi+1
        print(f"last evals: {evals_num}, testi {testi}")
        evals_num = run_test(testi, typid, exceed_evals=evals_num)


