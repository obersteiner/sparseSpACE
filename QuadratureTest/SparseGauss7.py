#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sys import path
path.append('../src/')
from Function import *
from StandardCombi import *
from Grid import *
import numpy as np
from mpi4py import MPI
from GridOperation import *
dim = 4
comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()
f1=VadereSimulation()
f2=FunctionPower(f1,2)
f=FunctionConcatenate([f1,f2])
if rank==0:
    a=np.array([0.1,-np.inf,-np.inf,1.3])
    b=np.array([0.3,np.inf,np.inf,1.8])
    grid1=GaussLegendreGrid1D(0.1, 0.3, boundary=False,normalize=True)   
    grid2=GaussHermiteGrid1D( boundary=False, loc=50, scale=1)
    grid3=GaussHermiteGrid1D( boundary=False, loc=60, scale=1)
    grid4=GaussLegendreGrid1D(1.3, 1.8, boundary=False,normalize=True)
    grids=[grid1,grid2,grid3,grid4]
  
    grid=MixedGrid(a=a,b=b,grids=grids)

    operation = Integration(f=f, grid=grid, dim=dim)
    combiObject = StandardCombi(a, b, operation=operation)
    minimum_level = 1
    maximum_level = 7
    scheme,smth,result,numpoints=combiObject.perform_operation(minimum_level, maximum_level)
    np.save("SparseGaussL7.npy",result)
    print(numpoints," points")
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

