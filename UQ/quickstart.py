##############################################################################
#
# This is a simple quickstart python program to show how to use chaospy
# to quantify the uncertainty of a simple function with a discontinuity.
#
# The goal is to create a sparse adaptive surrogate that can also be used
# instead of the model() function.
#
# Uncertain parameter:
# - parameter1
# - parameter2
# - parameter3
#
# Quantity of interest:
# - some value
#
# Author: Florian Kuenzner
#
##############################################################################


import chaospy as cp
import numpy as np
import math
import json
import os
from sys import path
path.append('../')
from Function import *
from spatiallyAdaptiveExtendSplit import *
from ErrorCalculator import *

#################################################################################################
# parameter setup
parameter1 = 0.3 # 0.3
parameter1_var = 0.03
parameter1_min = 0.1 #0.1
parameter1_max = 0.3 #0.5
parameter2 = 1.0
parameter2_var = 0.5 #0.03
parameter2_min = 1.0 #0.8
parameter2_max = 1.2 #1.2
parameter3 = 1.6
parameter3_var = 0.3
parameter3_min = 1.4 #1.4
parameter3_max = 1.8 #1.8

#################################################################################################
# setup uncertain parameter
parameter1Dist = cp.Uniform(parameter1_min, parameter1_max)
#parameter1Dist = cp.Normal(parameter1, parameter1_var)
parameter2Dist = cp.Uniform(parameter2_min, parameter2_max)
#parameter2Dist = cp.Normal(parameter2, parameter2_var)
parameter3 = cp.Uniform(parameter3_min, parameter3_max)

dist = cp.J(parameter1Dist, parameter2Dist, parameter3)

#################################################################################################
# generate nodes and weights
#q = 3  # number of collocation points for each dimension
#nodes, weights = cp.generate_quadrature(q, dist, rule="G")
model = FunctionUQ()
a = np.array([parameter1_min, parameter2_min, parameter3_min])
b = np.array([parameter1_max, parameter2_max, parameter3_max])
grid = GaussLegendreGrid(a,b,3)
errorOperator2=ErrorCalculatorExtendSplit()
adaptiveCombiInstanceExtend = SpatiallyAdaptiveExtendScheme(a, b,0,grid,version=0)
adaptiveCombiInstanceExtend.performSpatiallyAdaptiv(1,2,model,errorOperator2,10**-10, do_plot=False)
nodes, weights = adaptiveCombiInstanceExtend.get_points_and_weights()
nodes_transpose = list(zip(*nodes))

#################################################################################################
# propagate the uncertainty
value_of_interests = [model(node) for node in nodes]
value_of_interests = np.asarray(value_of_interests)

#################################################################################################
# generate orthogonal polynomials for the distribution
OP = cp.orth_ttr(3, dist)

#################################################################################################
# generate the general polynomial chaos expansion polynomial
gPCE = cp.fit_quadrature(OP, nodes_transpose, weights, value_of_interests)

#################################################################################################
# calculate statistics
E = cp.E(gPCE, dist)
StdDev = cp.Std(gPCE, dist)
first_order_sobol_indices = cp.Sens_m(gPCE, dist)
print(first_order_sobol_indices)
#print the stastics
print("mean: %f" % E)
print("stddev: %f" % StdDev)
