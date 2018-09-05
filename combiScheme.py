import numpy as np
import math

def getCombiScheme(lmin,lmax,dim, doPrint = True):
    gridArray = []
    for q in range(min(dim,lmax-lmin+1)):
        coefficient = (-1)**q * math.factorial(dim-1)/(math.factorial(q)*math.factorial(dim-1-q))
        grids = getGrids(dim,lmax-q)
        gridArray.extend([(np.array(g,dtype=int)+np.ones(dim,dtype=int)*(lmin-1),coefficient) for g in grids])
    for i in range(len(gridArray)):
        if(doPrint):
            print(i,list(gridArray[i][0]),gridArray[i][1])
    return gridArray

def getGrids(dimLeft,valuesLeft):
    if(dimLeft == 1):
        return [[valuesLeft]]
    grids = []
    for index in range(valuesLeft):
        levelvector = [index+1]
        grids.extend([levelvector + g  for g in getGrids(dimLeft-1,valuesLeft-index)])
    return grids