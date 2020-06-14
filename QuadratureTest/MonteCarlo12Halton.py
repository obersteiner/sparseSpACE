#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sys import path
path.append('../src/')
from Function import *
from mpi4py import MPI
#from Grid import *
#from GridOperation import *
import numpy as np
import chaospy as cp
#from spatiallyAdaptiveSingleDimension2 import *
#from ErrorCalculator import *
#import matplotlib.pyplot as plt
from numpy import linalg as LA
import MonteCarlo
import evalPoints
#import resource
#import time
#start=time.time()
comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()
f=VadereSimulation()
#f2=FunctionPower(f1,2)
if rank==0:

#f=FunctionConcatenate([f1,f2])
    dist2=cp.Normal(mu=50,sigma=1.0)
    dist4=cp.Uniform(1.3,1.8)
    dist3=cp.Normal(mu=60,sigma=1.0)
    dist1=cp.Uniform(0.1,0.3)
    dist=cp.J(dist1,dist2,dist3,dist4)
    path="Points"+str(int(28561**0.25) -1)+".npy"
    
    expected2=MonteCarlo.montecarloParallel(f,path)

    #output_dim = len(expected2) // 2
    #expectation = expected2[:output_dim]
   # expectation_of_squared = expected2[output_dim:]
    
    
    np.save("MCHalton12.npy",expected2)
    
    #usage=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    #print(usage,"usage")
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

# In[1]:





# In[ ]:




