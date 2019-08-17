'''
Created on 27.02.2015

Uncertainty quantification with the Stochastic collocation approach for a predator & prey model (Lotka & Voltera)

@author: Florian Kuenzner
'''

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


# Settings
timestep_problem = 25
uniform_distr = False
verbose = False


#predator = coyote
#prey = sheep

#initial parameters: sheep/coyote model!!
coyoteDeathRate = 0.0005 #death rate of coyote
sheepBirthRate = 0.005 #birth rate of sheep
voracity = 0.00012 #the voracity rate (when a predator meets sheep and kill it) (Gefraessigkeit)
augmentation = 0.002*voracity #the augmentation rate (when a coyote meets sheep and a new coyote growth) (Vermehrung)

sheeps_Px0 = 2000 #initial population size of sheep population
coyote_Px0 = 50 #initial population size of coyote population

T = 70*365 # end of simulation
NT = int(0.01 * T)  # number of time steps

# Standard deviations
sigma_voracity = 0.000002  # no uncertainty: 0.000000001, uncertainty: 0.000001
sigma_sheeps_Px0 = 1
sigma_coyote_Px0 = 5

# Maximum PCE polynomial degree
poly_deg_max = 1

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

# population model definition: as a initial value problem
def f(t, pX):
    '''
    ODE formulation of preyBirthRate predator & prey model.

    Parameters
    ----------
    pX : array[2], pX[0] is the population size of predator

                   pX[1] is the population size of prey
        Mean of the distribution.
    t : is the time

    f.predatorDeathRate : death rate of predator
    f.preyBirthRate : birth rate of prey
    f.voracity : the voracity rate (when predator meets prey and kill it)
    f.augmentation : the augmentation rate (when predator meets prey and a new predator growth)
    '''
    predatorPopulation, preyPopulation = pX

    predator = (-f.predatorDeathRate + f.augmentation*preyPopulation)*predatorPopulation
    prey = (f.preyBirthRate - f.voracity*predatorPopulation)*preyPopulation

    return [predator, prey]

time_points = np.linspace(0, T, NT+1)

def static_var(varname, value):
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate

@static_var("counter", 0)
def solver(voracity, Px0, f):
    #set the parameter
    f.preyBirthRate = sheepBirthRate
    f.predatorDeathRate = coyoteDeathRate
    f.voracity = voracity
    f.augmentation = augmentation

    #progress bar
    solver.counter += 1
    if solver.counter % 100 == 0:
        sys.stdout.write(".")

    #solves the population model
    #u = ode.odeint(f, Px0, time_points)
    #u = ode.solve_ivp(f, [0, T], Px0, method='BDF', t_eval=time_points)
    u = ode.solve_ivp(f, [0, T], Px0, method='RK45', t_eval=time_points)

    return u

def get_solver_value(input_values):
    # ~ voracity_sample, sheep_Px0_sample, coyote_Px0_sample = input_values
    voracity_sample, coyote_Px0_sample = input_values
    sheep_Px0_sample = sheeps_Px0
    if uniform_distr:
        # Chaospy uniform distribution does not work with small values
        voracity_sample = 0.000115 + 0.00001 * voracity_sample
        coyote_Px0_sample = 35 + coyote_Px0_sample * 30
    # y contains the predator solutions and prey solutions for all time values
    y = solver(voracity_sample, [coyote_Px0_sample, sheep_Px0_sample], f).y
    return y[1][timestep_problem]
problem_function = FunctionCustom(get_solver_value, output_dim=1)

error_operator = ErrorCalculatorSingleDimVolumeGuided()
op = UncertaintyQuantification(None, distris, a, b, dim=dim)
# ~ op = UncertaintyQuantification(problem_function, distris, a, b, dim=dim)
# ~ pa, pb = op.get_boundaries(0.01)
# ~ problem_function.plot(pa, pb, points_per_dim=5, filename="25.pdf")

types = ("Gauss", "adaptiveTrapez", "adaptiveHO", "Fejer", "adaptiveTransBSpline")
typids = dict()
for i,v in enumerate(types):
    typids[v] = i

# ~ i_ref = 256 + timestep_problem
if uniform_distr:
    E_pX_ref, P10_pX_ref, P90_pX_ref, Var_pX_ref = np.load("gauss_2D_uniform_solutions.npy")
else:
    # ~ E_pX_ref, P10_pX_ref, P90_pX_ref, Var_pX_ref = np.load("halton_2D_solutions.npy")
    E_pX_ref, P10_pX_ref, P90_pX_ref, Var_pX_ref = np.load("gauss_2D_solutions.npy")
assert len(Var_pX_ref) == 256
assert len(Var_pX_ref[0]) == 2
E_ref = E_pX_ref[timestep_problem][1]
P10_ref = P10_pX_ref[timestep_problem][1]
P90_ref = P90_pX_ref[timestep_problem][1]
Var_ref = Var_pX_ref[timestep_problem][1]

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
            grid = GlobalHighOrderGridWeighted(a, b, op, boundary=uniform_distr)
        elif typ in ("adaptiveTrapez", "Trapez"):
            grid = GlobalTrapezoidalGridWeighted(a, b, op, boundary=uniform_distr)
        elif typ == "adaptiveTransBSpline":
            grid = GlobalBSplineGrid(a_trans, b_trans, boundary=uniform_distr)
        elif typ == "adaptiveTransTrapez":
            grid = GlobalTrapezoidalGrid(a_trans, b_trans, boundary=uniform_distr)
        elif typ == "adaptiveTransHO":
            grid = GlobalHighOrderGrid(a_trans, b_trans, boundary=uniform_distr, split_up=False)

        if do_inverse_transform:
            # Use Integration operation
            f_refinement = op.get_inverse_transform_Function(op.get_PCE_Function(poly_deg_max))
            # ~ f_refinement.plot(np.array([0.001]*2), np.array([0.999]*2), filename="trans.pdf")
            op_integration = Integration(f_refinement, grid, dim)
            combiinstance = SpatiallyAdaptiveSingleDimensions2(a_trans, b_trans, operation=op_integration,
                norm=2, grid=grid)
        else:
            combiinstance = SpatiallyAdaptiveSingleDimensions2(a, b, operation=op,
                norm=2, grid=grid)
            f_refinement = op.get_PCE_Function(poly_deg_max)
        # ~ f_refinement = op.get_expectation_variance_Function()

        lmax = 3
        if typ == "Trapez":
            lmax = testi + 2
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
        op.calculate_PCE(None, combiinstance)
    else:
        polys, polys_norms = cp.orth_ttr(poly_deg_max, op.distributions_joint, retall=True)
        if typ == "Gauss":
            nodes, weights = cp.generate_quadrature(testi,
                op.distributions_joint, rule="G")
        elif typ == "Fejer":
            nodes, weights = cp.generate_quadrature(testi,
                op.distributions_joint, rule="F", normalize=True)
        f_evals = [problem_function_wrapped(c) for c in zip(*nodes)]
        op.gPCE = cp.fit_quadrature(polys, nodes, weights, np.asarray(f_evals), norms=polys_norms)

    print("simulation time: " + str(time.time() - measure_start) + " s")

    # ~ if False:
        # ~ E, var = op.calculate_expectation_and_variance(combiinstance)
        # ~ E_pX = reshape_result_values(E)
        # ~ Var = reshape_result_values(var)

    def reshape_result_values(vals): return vals[0]
    E = reshape_result_values(op.get_expectation_PCE())
    P10 = reshape_result_values(op.get_Percentile_PCE(10, 10*5))
    P90 = reshape_result_values(op.get_Percentile_PCE(90, 10*5))
    Var = reshape_result_values(op.get_variance_PCE())

    err_descs = ("E prey", "P10 prey", "P90 prey", "Var prey")
    err_data = (
        (E, E_ref),
        (P10, P10_ref),
        (P90, P90_ref),
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
    result_data = (num_evals, timestep_problem, typid, errors)
    assert len(result_data) == 4
    assert len(errors) == 8

    tmpdir = os.getenv("XDG_RUNTIME_DIR")
    results_path = tmpdir + "/uqtestSD.npy"
    solutions_data = []
    if os.path.isfile(results_path):
        solutions_data = list(np.load(results_path, allow_pickle=True))
    if all([any([d[i] != result_data[i] for i in range(3)]) for d in solutions_data]):
        solutions_data.append(result_data)
        np.save(results_path, solutions_data)

    return num_evals


# ~ evals_end = 900
evals_end = 400

# For testing
# ~ skip_types = ("Fejer", "adaptiveTransBSpline", "adaptiveTrapez", "Gauss", "adaptiveHO")
skip_types = ("Fejer", "adaptiveTransBSpline", "adaptiveTrapez")
assert all([typ in types for typ in skip_types])

for typid,typ in enumerate(types):
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


