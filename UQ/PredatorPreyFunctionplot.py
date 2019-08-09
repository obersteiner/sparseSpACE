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

# Create a Function that can be used for refining
def get_solver_values(input_values):
    # ~ voracity_sample, sheep_Px0_sample, coyote_Px0_sample = input_values
    voracity_sample, coyote_Px0_sample = input_values
    sheep_Px0_sample = sheeps_Px0
    # y contains the predator solutions and prey solutions for all time values
    y = solver(voracity_sample, [coyote_Px0_sample, sheep_Px0_sample], f).y
    return np.concatenate(y)
problem_function = FunctionCustom(get_solver_values, output_dim=len(time_points) * 2)

# This function is later required to bring calculated values into the right shape
def reshape_result_values(vals):
    mid = int(len(vals) / 2)
    predators, preys = vals[:mid], vals[mid:]
    return np.array([predators, preys]).T

op = UncertaintyQuantification(problem_function, distris, a, b, dim=dim)

pa, pb = op.get_boundaries(0.01)
tplen = len(time_points)
sheep_dims = np.linspace(tplen, 2 * tplen - 1, tplen, dtype=int)
# ~ sheep_dims = [sheep_dims[0], sheep_dims[1]]
HOME = os.getenv("HOME")
print(f"Plotting {len(sheep_dims)} result values")
problem_function.plot(pa, pb, points_per_dim=51, plotdimensions=sheep_dims,
    filename=HOME + "/tmp_mem/prey", consistent_axes=True, show_plot=False)
# for p in prey_*.png; do i=${p%.png}; i=${i#prey_}; i=$((i-256)); mv $p prey_${i}.png; done
# ffmpeg -f image2 -framerate 20 -i prey_%d.png  -c:v libvpx-vp9 -b:v 2M -pass 1 -an -f webm /dev/null
# ffmpeg -f image2 -framerate 20 -i prey_%d.png  -c:v libvpx-vp9 -b:v 2M -pass 2  output.webm
