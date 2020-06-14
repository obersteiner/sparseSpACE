#!/usr/bin/env python
# coding: utf-8

# In[1]:


#%matplotlib inline
from sys import path
path.append('../src/')
from Function import *
from mpi4py import MPI
from Grid import *
from GridOperation import *
import numpy as np
#import chaospy as cp
from spatiallyAdaptiveSingleDimension2 import *
from ErrorCalculator import *
#import matplotlib.pyplot as plt
#from numpy import linalg as LA
#import time
comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()
#start=time.time()
f=VadereSimulation()

if rank==0:

    sol_storage={}
    variance2=[]
    variance2Max=[]
    expectation2=[]
    expectation2Max=[]
    variance1=[]
    variance1Max=[]
    expectation1=[]
    expectation1Max=[]
    nodes=[]
    Var_to_save=[]
    E_to_save=[]
    Err_to_save=[]
    Errsq_to_save=[]    
    distributions=[("Uniform", 0.1, 0.3), ("Normal", 50, 1), ("Normal", 60, 1), ("Uniform", 1.3, 1.8)]
    a=np.array([0.1,-np.inf,-np.inf,1.3])
    b=np.array([0.3,np.inf,np.inf,1.8])
    op = UncertaintyQuantification(f, distributions, a, b)
    grid = GlobalTrapezoidalGridWeighted(a, b, op, boundary=False,modified_basis=False)
    op.set_grid(grid)

    op.set_expectation_variance_Function()

    combiinstance = SpatiallyAdaptiveSingleDimensions2(a, b, operation=op, norm=2,version=6)
    lmax = 2
    error_operator = ErrorCalculatorSingleDimVolumeGuided()
    refinement,scheme,lmax,cimbires,numevals,error,numpoint,surpluserror,errorL2,errormax,Nerr=combiinstance.performSpatiallyAdaptiv(1, lmax,error_operator, tol=0, max_evaluations=20000, do_plot=False,solutions_storage=sol_storage)
    combiinstance.draw_refinement(filename="graphs/RefinementGraph5L2Ver6.pdf")

        
   
                
    np.save('SolStorageVer6_20k.npy',sol_storage)    
        
    np.save('AdaptiveErrorVer6_20k.npy',Nerr)

    for i in range(size-1):  
                  comm.send(1,dest=i+1)
data=[]       
if rank>0:
    while isinstance(data,list):
        
            data=comm.recv(source=0)
            if isinstance(data,int):
                break
            else:    
                if (len(data)>0):
                        for i in range(len(data)):
                                  f.eval(data[i])
                comm.send(True,dest=0)                




# In[ ]:




