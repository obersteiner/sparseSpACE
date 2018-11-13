#%matplotlib inline
import matplotlib.pyplot as plt
from sys import path
import numpy as np

#from numpy import *
import scipy as sp
from spatiallyAdaptiveExtendSplit import *
from spatiallyAdaptiveSplit import *
from spatiallyAdaptiveSingleDimension import *
from spatiallyAdaptiveCell import *

from PerformTestCase import *
from Function import *
from ErrorCalculator import *
import math

a = 0
b = 1
dim = 2
midpoint = np.ones(dim) * 0.5
coefficients = np.array([ 10**0 * (d+1) for d in range(dim)])
f = GenzDiscontinious(border=midpoint,coeffs=coefficients)
f.plot(np.ones(dim)*a,np.ones(dim)*b)
errorOperator=ErrorCalculatorSurplusCell()
errorOperator2=ErrorCalculatorAnalytic()

grid=TrapezoidalGrid(np.ones(dim)*a, np.ones(dim)*b)
adaptiveCombiInstanceSingleDim = SpatiallyAdaptivSingleDimensions(np.ones(dim)*a, np.ones(dim)*b,grid)
adaptiveCombiInstanceFixed = SpatiallyAdaptivFixedScheme(np.ones(dim)*a, np.ones(dim)*b,grid)
adaptiveCombiInstanceExtend2 = SpatiallyAdaptivExtendScheme(np.ones(dim)*a, np.ones(dim)*b,1,grid,False,True)
adaptiveCombiInstanceExtend = SpatiallyAdaptivExtendScheme(np.ones(dim)*a, np.ones(dim)*b,2,grid,False,True)
adaptiveCombiInstanceCell = SpatiallyAdaptivCellScheme(np.ones(dim)*a, np.ones(dim)*b,grid)

adaptiveCombiInstanceCell.performSpatiallyAdaptiv(2,2,f,errorOperator,10**-2, do_plot=True)

adaptiveCombiInstanceExtend.performSpatiallyAdaptiv(1,2,f,errorOperator2,10**-2, do_plot=True)

