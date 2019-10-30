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
do_HighOrder = False
# ~ do_HighOrder = True
lmax = 3
if "max_evals" in os.environ:
    max_evals = int(os.environ["max_evals"])
else:
    max_evals = 150
# Use reference values to calculate errors
calculate_errors = True
# ~ calculate_errors = False
save_errors = True


#predator = coyote
#prey = sheep

voracity = 0.00012 #the voracity rate (when a predator meets sheep and kill it) (Gefraessigkeit)

# ~ sheeps_Px0 = 2000 #initial population size of sheep population
coyote_Px0 = 50 #initial population size of coyote population

# Standard deviations
sigma_voracity = 0.000002  # no uncertainty: 0.000000001, uncertainty: 0.000001
# ~ sigma_sheeps_Px0 = 1
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

measure_start = time.time()

print("Generating quadrature nodes and weights")

T = pp.T
time_points = pp.time_points

def get_solver_values(input_values):
    return np.concatenate(pp.get_solver_value2D(input_values))
problem_function = FunctionCustom(get_solver_values, output_dim=len(time_points)*2)

# This function is later required to bring calculated values into the right shape
def reshape_result_values(vals):
    mid = int(len(vals) / 2)
    predators, preys = vals[:mid], vals[mid:]
    return np.array([predators, preys]).T

# Create the Operation
op = UncertaintyQuantification(problem_function, distris, a, b, dim=dim)
# ~ pa, pb = op.get_boundaries(0.01)
# ~ problem_function.plot(pa, pb, points_per_dim=31, plotdimension=410)

#'''
# Do the spatially adaptive refinement for PCE
error_operator = ErrorCalculatorSingleDimVolumeGuided()
# ~ error_operator = ErrorCalculatorSingleDimVolumeGuidedPunishedDepth()
if do_HighOrder:
    grid = GlobalHighOrderGridWeighted(a, b, op, boundary=False, modified_basis=False)
else:
    grid = GlobalTrapezoidalGridWeighted(a, b, op, boundary=False)
op.set_grid(grid)
combiinstance = SpatiallyAdaptiveSingleDimensions2(a, b, operation=op,
    norm=2)
# ~ combiinstance = StandardCombi(a, b, operation=op, grid=grid)
tol = 0
f_refinement = op.get_PCE_Function(poly_deg_max)
# ~ f_refinement = ConstantValue(1.0)
combiinstance.performSpatiallyAdaptiv(1, lmax, f_refinement, error_operator, tol=tol,
    max_evaluations=max_evals, print_output=not silent_mode, do_plot=False)
# ~ op.grid=grid
# ~ combiinstance.perform_combi(1, lmax, f_refinement)
if False:
    f_exvar = op.get_expectation_variance_Function()
    combiinstance.performSpatiallyAdaptiv(1, 2, f_exvar, error_operator, tol=tol,
        max_evaluations=max_evals)

# Calculate the gPCE using the nodes and weights from the refinement
# ~ op.calculate_PCE(poly_deg_max, combiinstance, use_combiinstance_solution=False)
op.calculate_PCE(poly_deg_max, combiinstance)
if not silent_mode:
    print("simulation time: " + str(time.time() - measure_start) + " s")
'''
# Gauss quadrature
op.calculate_PCE_chaospy(poly_deg_max, 10)
print("non-sparsegrid simulation time: " + str(time.time() - measure_start) + " s")
#'''

if False:
    E, var = op.calculate_expectation_and_variance(combiinstance)
    E_pX = reshape_result_values(E)
    Var = reshape_result_values(var)

##extract the statistics
# expectation value
E_pX = reshape_result_values(op.get_expectation_PCE())
# percentiles
P10_pX = reshape_result_values(op.get_Percentile_PCE(10, 10*5))
P90_pX = reshape_result_values(op.get_Percentile_PCE(90, 10*5))
# variance
Var = reshape_result_values(op.get_variance_PCE())


if calculate_errors:
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

    if save_errors:
        num_evals = problem_function.get_f_dict_size()
        tmpdir = os.getenv("XDG_RUNTIME_DIR")
        results_path = tmpdir + "/uqtest.npy"
        solutions_data = []
        if os.path.isfile(results_path):
            solutions_data = list(np.load(results_path, allow_pickle=True))
        solutions_data.append((num_evals, mean_errs))
        np.save(results_path, solutions_data)


if silent_mode:
    sys.exit()

#plot the stuff
time_points = time_points/365
figure = plotter.figure(1, figsize=(13,10))
figure.canvas.set_window_title('Stochastic Collocation: Coyote, Sheep (Predator, Prey)')

#sheep expectation value
plotter.subplot(421)
plotter.title('Sheep Expectation')
plotter.plot(time_points, E_pX.T[1], label='E Sheep')
plotter.fill_between(time_points, P10_pX.T[1], P90_pX.T[1], facecolor='#5dcec6')
plotter.plot(time_points, P10_pX.T[1], label='P10')
plotter.plot(time_points, P90_pX.T[1], label='P90')
plotter.xlabel('time (t) - years')
plotter.ylabel('population size')
plotter.xlim(0, T/365)
plotter.legend(loc=2) #enable the legend
plotter.grid(True)

#coyote expectation value
plotter.subplot(423)
plotter.title('Coyote Expectation')
plotter.plot(time_points, E_pX.T[0], label='E Coyote')
plotter.fill_between(time_points, P10_pX.T[0], P90_pX.T[0], facecolor='#5dcec6')
plotter.plot(time_points, P10_pX.T[0], label='P10')
plotter.plot(time_points, P90_pX.T[0], label='P90')
plotter.xlabel('time (t) - years')
plotter.ylabel('population size')
plotter.xlim(0, T/365)
plotter.legend(loc=2) #enable the legend
plotter.grid(True)

#sheep variance
plotter.subplot(422)
plotter.title('Sheep Variance')
plotter.plot(time_points, Var.T[1], label="Sheep")
plotter.xlabel('time (t) - years')
plotter.ylabel('variance')
plotter.legend(loc=2) #enable the legend
plotter.xlim(0, T/365)
plotter.grid(True)

#coyote variance
plotter.subplot(424)
plotter.title('Coyote Variance')
plotter.plot(time_points, Var.T[0], label="Coyote")
plotter.xlabel('time (t) - years')
plotter.ylabel('variance')
plotter.xlim(0, T/365)
plotter.legend(loc=2) #enable the legend
plotter.grid(True)


if calculate_errors:
    def plot_error(pos, descr_and_data):
        plotter.subplot(pos)
        for descr, data in descr_and_data:
            plotter.plot(time_points, data, label=descr + ' error')
        # ~ plotter.yscale("log")
        plotter.xlim(0, T/365)
        plotter.legend(loc=2)
        plotter.grid(True)

    plot_error(425, [("Sheep E_pX relative", error_E_predator), ("Coyote E_pX relative", error_E_prey)])
    # ~ plot_error(426, [("Sheep Var", error_Var_prey), ("Coyote Var", error_Var_predator)])
    plot_error(426, [("Sheep Var", error_Var_prey)])
    plot_error(427, [("Sheep P10", error_P10_prey), ("Sheep P90", error_P90_prey)])
    plot_error(428, [("Coyote P10", error_P10_predator), ("Coyote P90", error_P90_predator)])
    # ~ plot_error(427, [("Sheep P10", error_P10_prey), ("Coyote P10", error_P10_predator)])
    # ~ plot_error(428, [("Sheep P90", error_P90_prey), ("Coyote P90", error_P90_predator)])


#save figure
fileName = os.path.splitext(sys.argv[0])[0] + '.pdf'
plotter.savefig(fileName, format='pdf')

plotter.show()

