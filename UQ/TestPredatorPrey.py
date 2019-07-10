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
NT = 0.01*T # no of time steps

voracity_dist   = cp.Normal(voracity,   0.000002) #no uncertainty: 0.000000001, uncertainty: 0.000001
sheep_Px0_dist  = cp.Normal(sheeps_Px0, 1) #no uncertainty: 0.000000001, uncertainty: 250, 100, 50, 25
coyote_Px0_dist = cp.Normal(coyote_Px0, 0.1) #no uncertainty: 0.000000001, uncertainty: 2, 1, 0.5

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

#generate the joined distribution
dist = cp.J(voracity_dist, sheep_Px0_dist, coyote_Px0_dist);
#generate the nodes and weights
nodes, weights = cp.generate_quadrature(10, dist)

#solve for each collocation node
sol = [solver(voracity_sample, [coyote_Px0_sample, sheep_Px0_sample], f)
             for voracity_sample, sheep_Px0_sample, coyote_Px0_sample in nodes.T]
samples_u = [s.y.T for s in sol]

P = cp.orth_ttr(1, dist)
# do the stochastic collocation simulation:
coefficents_P = cp.fit_quadrature(P, nodes, weights, samples_u)

measure_end = time.time()
print("simulation time: " + str(measure_end - measure_start) + " sec")

##extract the statistics
# expectation value
E_pX = cp.E(coefficents_P, dist)
# percentiles
P10_pX = cp.Perc(coefficents_P, 10, dist, 10*5)
P90_pX = cp.Perc(coefficents_P, 90, dist, 10*5)
# variance
Var = cp.Var(coefficents_P, dist)

#plot the stuff
time_points = time_points/365
figure = plotter.figure(1, figsize=(13,5))
figure.canvas.set_window_title('Stochastic Collocation: Coyote, Sheep (Predator, Prey)')

#sheep expectation value
plotter.subplot(221)
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
plotter.subplot(223)
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
plotter.subplot(222)
plotter.title('Sheep (Var)')
plotter.plot(time_points, Var.T[1], label="Sheep")
plotter.xlabel('time (t) - years')
plotter.ylabel('variance')
plotter.legend(loc=2) #enable the legend
plotter.xlim(0, T/365)
plotter.grid(True)

#coyote variance
plotter.subplot(224)
plotter.title('Coyote (Var)')
plotter.plot(time_points, Var.T[0], label="Coyote")
plotter.xlabel('time (t) - years')
plotter.ylabel('variance')
plotter.xlim(0, T/365)
plotter.legend(loc=2) #enable the legend
plotter.grid(True)

#save figure
fileName = os.path.splitext(sys.argv[0])[0] + '.pdf'
plotter.savefig(fileName, format='pdf')

plotter.show()

