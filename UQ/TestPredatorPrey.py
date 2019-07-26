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
sys.path.append('../')
from Function import *
from spatiallyAdaptiveSingleDimension2 import *
from ErrorCalculator import *
from GridOperation import *

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
sigma_sheeps_Px0 = 1.0  # no uncertainty: 0.000000001, uncertainty: 250, 100, 50, 25
sigma_coyote_Px0 = 0.1  # no uncertainty: 0.000000001, uncertainty: 2, 1, 0.5

# Maximum PCE polynomial degree
poly_deg_max = 1

# Distributions information to be passed to the UncertaintyQuantification Operation
distris = [
    ("Normal", voracity, sigma_voracity),
    ("Normal", sheeps_Px0, sigma_sheeps_Px0),
    ("Normal", coyote_Px0, sigma_coyote_Px0)
]
dim = len(distris)
# Normal distribution requires infinite boundaries
a = np.array([-np.inf for _ in range(dim)])
b = np.array([np.inf for _ in range(dim)])
# Settings for the adaptive refinement quad weights
do_HighOrder = False
# ~ do_HighOrder = True
# ~ max_evals = 10 ** dim
max_evals = 500
# Use reference values to calculate errors
calculate_errors = True
# ~ calculate_errors = False

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

measure_start = time.time()

print("Generating quadrature nodes and weights")

# Create a Function that can be used for refining
def get_solver_values(input_values):
    voracity_sample, sheep_Px0_sample, coyote_Px0_sample = input_values
    # y contains the predator solutions and prey solutions for all time values
    y = solver(voracity_sample, [coyote_Px0_sample, sheep_Px0_sample], f).y
    return np.concatenate(y)
problem_function = FunctionCustom(get_solver_values)

# This function is later required to bring calculated values into the right shape
def reshape_result_values(vals):
    mid = int(len(vals) / 2)
    predators, preys = vals[:mid], vals[mid:]
    return np.array([predators, preys]).T

# Create the Operation
op = UncertaintyQuantification(problem_function, distris, a, b, dim=dim)

#'''
# Do the spatially adaptive refinement for PCE
error_operator = ErrorCalculatorSingleDimVolumeGuided()
# ~ error_operator = ErrorCalculatorSingleDimVolumeGuidedPunishedDepth()
combiinstance = SpatiallyAdaptiveSingleDimensions2(a, b, operation=op,
    boundary=False, norm=2, do_high_order=do_HighOrder)
tol = 10 ** -4
f_pce = op.get_PCE_Function(poly_deg_max)
combiinstance.performSpatiallyAdaptiv(1, 2, f_pce, error_operator, tol=tol,
    max_evaluations=max_evals)
if False:
    f_exvar = op.get_expectation_variance_Function()
    combiinstance.performSpatiallyAdaptiv(1, 2, f_exvar, error_operator, tol=tol,
        max_evaluations=max_evals)

# Calculate the gPCE using the nodes and weights from the refinement
op.calculate_PCE(poly_deg_max, combiinstance)
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
    E_pX_halton, P10_pX_halton, P90_pX_halton, Var_pX_halton = np.load("halton_solutions.npy")
    E_predator, E_prey = E_pX.T
    P10_predator, P10_prey = P10_pX.T
    P90_predator, P90_prey = P90_pX.T
    Var_predator, Var_prey = Var.T
    def calc_error(vals, reference_vals):
        return np.array([abs((vals[i] - sol) / sol) if not isclose(sol, 0.0) else abs(vals[i]) for i,sol in enumerate(reference_vals)])
    error_E_predator = calc_error(E_predator, E_pX_halton.T[0])
    error_E_prey = calc_error(E_prey, E_pX_halton.T[1])
    error_P10_predator = calc_error(P10_predator, P10_pX_halton.T[0])
    error_P10_prey = calc_error(P10_prey, P10_pX_halton.T[1])
    error_P90_predator = calc_error(P90_predator, P90_pX_halton.T[0])
    error_P90_prey = calc_error(P90_prey, P90_pX_halton.T[1])
    error_Var_predator = calc_error(Var_predator, Var_pX_halton.T[0])
    error_Var_prey = calc_error(Var_prey, Var_pX_halton.T[1])

    def mean_error(data):
        return np.sum(data) / len(data)
    mean_errs = (
        mean_error(error_E_prey), mean_error(error_E_predator),
        mean_error(error_P10_prey), mean_error(error_P10_predator),
        mean_error(error_P90_prey), mean_error(error_P90_predator),
        mean_error(error_Var_prey), mean_error(error_Var_predator)
    )
    mean_err_descs = ("E prey", "E predator", "P10 prey", "P10 predator",
        "P90 prey", "P90 predator", "Var prey", "Var predator")
    for i,desc in enumerate(mean_err_descs):
        print(f"{desc} mean relative error: {mean_errs[i]:.5g}")


#plot the stuff
time_points = time_points/365
figure = plotter.figure(1, figsize=(13,10))
figure.canvas.set_window_title('Stochastic Collocation: Coyote, Sheep (Predator, Prey)')

#sheep expectation value
plotter.subplot(421)
plotter.title('Sheep (E_pX)')
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
plotter.title('Coyote (E_pX)')
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
plotter.title('Sheep (Var)')
plotter.plot(time_points, Var.T[1], label="Sheep")
plotter.xlabel('time (t) - years')
plotter.ylabel('variance')
plotter.legend(loc=2) #enable the legend
plotter.xlim(0, T/365)
plotter.grid(True)

#coyote variance
plotter.subplot(424)
plotter.title('Coyote (Var)')
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
            plotter.plot(time_points, data, label=descr + ' relative error')
        # ~ plotter.yscale("log")
        plotter.xlim(0, T/365)
        plotter.legend(loc=2)
        plotter.grid(True)

    plot_error(425, [("Sheep E_pX", error_E_predator), ("Coyote E_pX", error_E_prey)])
    plot_error(426, [("Sheep Var", error_Var_prey), ("Coyote Var", error_Var_predator)])
    plot_error(427, [("Sheep P10", error_P10_prey), ("Sheep P90", error_P90_prey)])
    plot_error(428, [("Coyote P10", error_P10_predator), ("Coyote P90", error_P90_predator)])
    # ~ plot_error(427, [("Sheep P10", error_P10_prey), ("Coyote P10", error_P10_predator)])
    # ~ plot_error(428, [("Sheep P90", error_P90_prey), ("Coyote P90", error_P90_predator)])


#save figure
fileName = os.path.splitext(sys.argv[0])[0] + '.pdf'
plotter.savefig(fileName, format='pdf')

plotter.show()

