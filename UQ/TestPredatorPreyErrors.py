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
# ~ silent_mode = True
silent_mode = False

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
distris = [
    ("Normal", voracity, sigma_voracity),
    # ~ ("Normal", sheeps_Px0, sigma_sheeps_Px0),
    ("Normal", coyote_Px0, sigma_coyote_Px0)
]
dim = len(distris)
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
def solver(voracity, Px0, f, time_pts):
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
    u = ode.solve_ivp(f, [0, T], Px0, method='RK45', t_eval=time_pts)

    return u

measure_start = time.time()

print("Generating quadrature nodes and weights")

def get_solver_values(input_values):
    # ~ voracity_sample, sheep_Px0_sample, coyote_Px0_sample = input_values
    voracity_sample, coyote_Px0_sample = input_values
    sheep_Px0_sample = sheeps_Px0
    # y contains the predator solutions and prey solutions for all time values
    y = solver(voracity_sample, [coyote_Px0_sample, sheep_Px0_sample], f, time_points).y
    return np.concatenate(y)
problem_function = FunctionCustom(get_solver_values)

# This function is later required to bring calculated values into the right shape
def reshape_result_values(vals):
    mid = int(len(vals) / 2)
    predators, preys = vals[:mid], vals[mid:]
    return np.array([predators, preys]).T

use_proxy = False
if use_proxy:
    time_points_proxy = np.linspace(0, T, 31)
    # ~ time_points_proxy = time_points
    def get_solver_values_proxy(input_values):
        voracity_sample, sheep_Px0_sample, coyote_Px0_sample = input_values
        # y contains the predator solutions and prey solutions for all time values
        y = solver(voracity_sample, [coyote_Px0_sample, sheep_Px0_sample], f, time_points_proxy).y
        return np.concatenate(y)
    proxy_function = FunctionCustom(get_solver_values_proxy)


# Create the Operation
op = UncertaintyQuantification(None, distris, a, b, dim=dim)

types = ("Gauss", "adaptiveTrapez", "adaptiveHO", "Trapez")

def run_test(testi, typid, exceed_evals=None):
    if use_proxy:
        proxy_function_wrapped = FunctionCustom(lambda x: proxy_function(x))
    problem_function_wrapped = FunctionCustom(lambda x: problem_function(x))
    op.f = problem_function_wrapped

    typ = types[typid]
    if typ != "Gauss":
        if typ == "adaptiveHO":
            grid = GlobalHighOrderGridWeighted(a, b, op, boundary=False, modified_basis=False)
        elif typ in ("adaptiveTrapez", "Trapez"):
            grid = GlobalTrapezoidalGridWeighted(a, b, op, boundary=False)
        combiinstance = SpatiallyAdaptiveSingleDimensions2(a, b, operation=op,
            norm=2, grid=grid)
        tol = 10 ** -4
        error_operator = ErrorCalculatorSingleDimVolumeGuided()
        # ~ f_pce = op.get_PCE_Function(poly_deg_max)
        if use_proxy:
            op.f = proxy_function_wrapped
            f_refinement = op.get_expectation_variance_Function()
            op.f = -1
        else:
            f_refinement = op.get_expectation_variance_Function()

        lmax = 3
        if typ == "Trapez":
            lmax = testi + 2
        if exceed_evals is None or typ == "Trapez":
            combiinstance.performSpatiallyAdaptiv(1, lmax, f_refinement,
                error_operator, tol=0,
                max_evaluations=1,
                print_output=False)
        else:
            combiinstance.performSpatiallyAdaptiv(1, lmax, f_refinement,
                error_operator, tol=np.inf,
                max_evaluations=np.inf, min_evaluations=exceed_evals+1,
                print_output=False)

        # Calculate the gPCE using the nodes and weights from the refinement
        op.f = problem_function_wrapped
        op.calculate_PCE(poly_deg_max, combiinstance, use_combiinstance_solution=False)
    else:
        op.calculate_PCE_chaospy(poly_deg_max, testi+1)

    ##extract the statistics
    # expectation value
    E_pX = reshape_result_values(op.get_expectation_PCE())
    # percentiles
    P10_pX = reshape_result_values(op.get_Percentile_PCE(10, 10*5))
    P90_pX = reshape_result_values(op.get_Percentile_PCE(90, 10*5))
    # variance
    Var = reshape_result_values(op.get_variance_PCE())

    # ~ E_pX_ref, P10_pX_ref, P90_pX_ref, Var_pX_ref = np.load("gauss_solutions.npy")
    E_pX_ref, P10_pX_ref, P90_pX_ref, Var_pX_ref = np.load("gauss_2D_solutions.npy")
    E_predator, E_prey = E_pX.T
    P10_predator, P10_prey = P10_pX.T
    P90_predator, P90_prey = P90_pX.T
    Var_predator, Var_prey = Var.T
    def calc_error(vals, reference_vals):
        return np.array([abs(vals[i] - sol) for i,sol in enumerate(reference_vals)])
    def calc_error_relative(vals, reference_vals):
        errs = calc_error(vals, reference_vals)
        return np.array([abs(errs[i] / sol) if not isclose(sol, 0.0) else errs[i] for i,sol in enumerate(reference_vals)])
    error_E_predator = calc_error_relative(E_predator, E_pX_ref.T[0])
    error_E_prey = calc_error_relative(E_prey, E_pX_ref.T[1])
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
        mean_error(error_E_prey), mean_error(error_E_predator),
        mean_error(error_P10_prey), mean_error(error_P10_predator),
        mean_error(error_P90_prey), mean_error(error_P90_predator),
        mean_error(error_Var_prey), mean_error(error_Var_predator)
    )
    if not silent_mode:
        mean_err_descs = ("E prey", "E predator", "P10 prey", "P10 predator",
            "P90 prey", "P90 predator", "Var prey", "Var predator")
        for i,desc in enumerate(mean_err_descs):
            print(f"{desc} mean error: {mean_errs[i]:.5g}")

    num_evals = problem_function_wrapped.get_f_dict_size()
    # ~ num_evals = proxy_function_wrapped.get_f_dict_size()
    result_data = (num_evals, typid, mean_errs)
    assert len(result_data) == 3

    tmpdir = os.getenv("XDG_RUNTIME_DIR")
    results_path = tmpdir + "/uqtest.npy"
    solutions_data = []
    if os.path.isfile(results_path):
        solutions_data = list(np.load(results_path, allow_pickle=True))
    if all([any([d[i] != result_data[i] for i in range(2)]) for d in solutions_data]):
        solutions_data.append(result_data)
        np.save(results_path, solutions_data)

    if use_proxy:
        return proxy_function_wrapped.get_f_dict_size()
    return num_evals


evals_end = 200

for typid,typ in enumerate(types):
    print("Calculations for", typ)
    testi = 0
    evals_num = run_test(testi, typid)
    while evals_num < evals_end:
        testi = testi+1
        print(f"last evals: {evals_num}, testi {testi}")
        evals_num = run_test(testi, typid, exceed_evals=evals_num)

# ~ print("Calculating convent. errors")
# ~ for i in range(1, math.ceil(evals_end ** (1/dim))):
    # ~ print("order: ", i)
    # ~ # Gauss
    # ~ run_test(i, 0)


