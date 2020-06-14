#!/usr/bin/env python
# coding: utf-8

# In[2]:


#%matplotlib inline
from sys import path
#path.append('../src/')
from Function import *
from mpi4py import MPI
#from Grid import *
#from GridOperation import *
import numpy as np
import chaospy as cp
import matplotlib.pyplot as plt
from numpy import linalg as LA
import MonteCarlo
import evalPoints
comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()
#the test with 12^4 evaluation points was calculated in two tests: for the functions f1 and f1^2 two files were saved with the respective integrals
f1=VadereSimulation()
f2=FunctionPower(f1,2)
f=FunctionConcatenate([f1,f2])
if rank==0:
    dist2=cp.Normal(mu=50,sigma=1.0)
    dist4=cp.Uniform(1.3,1.8)
    dist3=cp.Normal(mu=60,sigma=1.0)
    dist1=cp.Uniform(0.1,0.3)
    a=np.array([0.1,-np.inf,-np.inf,1.3])
    b=np.array([0.3,np.inf,np.inf,1.8])
    dist=cp.J(dist1,dist2,dist3,dist4)
    P=cp.orth_ttr(1,dist)

    abscissas,weights=cp.generate_quadrature(order=(11),dist=dist,rule="Gaussian")

    new_points=[tuple(coord) for coord in abscissas.T]
    evalPoints.start_parallel_evaluation(f,list(new_points))

            
    evals=[f(coord) for coord in abscissas.T]
    f_approx=cp.fit_quadrature(P, abscissas,weights,evals)
    expected2=cp.E(f_approx,dist)
    output_dim = len(expected2) // 2
    expectation = expected2[:output_dim]
    expectation_of_squared = expected2[output_dim:]
    np.save("Gauss11ProdRunE.npy",[expectation,expectation_of_squared])
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
                                  f(data[i])
                comm.send(True,dest=0)                







