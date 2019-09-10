import numpy as np
import chaospy as cp
import os

import sys
sys.path.append('../src/')
from Function import *
from spatiallyAdaptiveSingleDimension2 import *
from ErrorCalculator import *
from GridOperation import *


d = 2
shifted = True
verbose = False

types = ("Gauss", "adaptiveTrapez", "adaptiveHO", "BSpline", "adaptiveLagrange", "sparseGauss", "adaptiveTrapezMB")

a = np.zeros(d)
b = np.ones(d)
if shifted:
    f_g = FunctionGShifted(d)
else:
    f_g = FunctionG(d)
reference_expectation = f_g.get_expectation()
reference_variance = f_g.get_variance()
# Create the operation only once
op = UncertaintyQuantificationTesting(None, "Uniform", a, b, dim=d)

def run_test(typi, typid, exceed_evals=None, evals_end=None, max_time=None):

    f = FunctionCustom(lambda x: f_g(x))
    op.f = f

    multiple_evals = None
    typ = types[typid]
    lmax = 3 if typ == "adaptiveTrapezMB" else 2
    if typ not in ("Gauss", "sparseGauss"):
        if typ == "adaptiveHO":
            if False:
                # A non-weighted grid can be used due to the uniform distribution
                # This currently does not work.
                assert all([v == 0 for v in a])
                assert all([v == 1 for v in b])
                grid = GlobalHighOrderGrid(a, b, boundary=boundary, modified_basis=modified_basis, split_up=False)
            else:
                grid = GlobalHighOrderGridWeighted(a, b, op, boundary=True)
        elif typ == "adaptiveTrapez":
            grid = GlobalTrapezoidalGridWeighted(a, b, op, boundary=True)
        elif typ == "adaptiveTrapezMB":
            grid = GlobalTrapezoidalGridWeighted(a, b, op, boundary=False, modified_basis=True)
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
        # ~ expectation_var_func.plot(a, b, filename="exp.pdf", plotdimension=0)
        # ~ expectation_var_func.plot(a, b, filename="mom2.pdf", plotdimension=1)
        mom2 = reference_variance + reference_expectation * reference_expectation
        reference_solution = np.array([reference_expectation, mom2])
        op.set_reference_solution(reference_solution)
        if evals_end is not None:
            multiple_evals = dict()
            combiinstance.performSpatiallyAdaptiv(1, lmax, expectation_var_func,
                error_operator, tol=0, max_evaluations=evals_end,
                print_output=True, solutions_storage=multiple_evals,
                max_time=max_time)
        elif exceed_evals is None:
            combiinstance.performSpatiallyAdaptiv(1, lmax, expectation_var_func,
                error_operator, tol=0,
                max_evaluations=1,
                print_output=verbose)
        else:
            combiinstance.performSpatiallyAdaptiv(1, lmax, expectation_var_func,
                error_operator, tol=np.inf,
                max_evaluations=np.inf, min_evaluations=exceed_evals+1,
                print_output=verbose)

        if multiple_evals is None:
            (E,), (Var,) = op.calculate_expectation_and_variance(combiinstance)
    else:
        if typ == "Gauss":
            nodes, weights = cp.generate_quadrature(typi, op.distributions_joint, rule="G")
        elif typ == "sparseGauss":
            op.set_grid(GaussLegendreGrid(a, b))
            combiinstance = StandardCombi(a, b, operation=op)
            combiinstance.perform_combi(1, testi+1, op.get_expectation_variance_Function())
            nodes, weights = combiinstance.get_points_and_weights()
            nodes = nodes.T
        E, Var = op.calculate_expectation_and_variance_for_weights(nodes, weights)

    # ~ print(f"E: {E}, Var: {Var}\n")
    # ~ print("reference E and Var: ", reference_expectation, reference_variance)

    tmpdir = os.getenv("XDG_RUNTIME_DIR")
    results_path = tmpdir + "/uqtestG.npy"
    solutions_data = []
    if os.path.isfile(results_path):
        solutions_data = list(np.load(results_path, allow_pickle=True))

    if multiple_evals is None:
        err_E = abs((E - reference_expectation) / reference_expectation)
        err_Var = abs((Var - reference_variance) / reference_variance)
        num_evals = f.get_f_dict_size()

        print("evals, relative errors:", num_evals, err_E, err_Var)

        result_data = (num_evals, typid, err_E, err_Var)

        if all([any([d[i] != result_data[i] for i in range(2)]) for d in solutions_data]):
            solutions_data.append(result_data)
            np.save(results_path, solutions_data)

        return num_evals
    else:
        solutions = op.calculate_multiple_expectation_and_variance(multiple_evals)
        for num_evals, E, Var in solutions:
            err_E = abs((E - reference_expectation) / reference_expectation)
            err_Var = abs((Var - reference_variance) / reference_variance)

            print("evals, relative errors:", num_evals, err_E, err_Var)

            result_data = (num_evals, typid, err_E, err_Var)

            if all([any([d[i] != result_data[i] for i in range(2)]) for d in solutions_data]):
                solutions_data.append(result_data)
        np.save(results_path, solutions_data)
        return f.get_f_dict_size()


evals_end = 4000
# ~ evals_end = 25000
max_time = 30
# ~ max_time = 560

# For testing
# ~ skip_types = ("sparseGauss", "adaptiveLagrange", "BSpline", "adaptiveHO")
skip_types = ()
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
