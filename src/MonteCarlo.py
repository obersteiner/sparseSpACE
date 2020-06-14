import numpy as np
import random
from mpi4py import MPI
import evalPoints
import chaospy as cp
#this method defines a basic monte carlo integration by evaluation random points
def montecarlo(f,N,dim, a, b):
    position = np.zeros(dim)
    result = 0
    for n in range(N):
        for d in range(dim):
            position[d] = random.random()*(b[d] - a[d]) + a[d]
        result +=f.eval(position)
    return result/N

def montecarloParallel(f,path_to_points):   
    result = 0    
    points=[]
    new_points=set()                                   
    points=np.load(path_to_points,allow_pickle=True)        
    for position in points:
            new_points.add(tuple(position))              
    evalPoints.start_parallel_evaluation(f,list(new_points))                
    for position in points:
                result +=f(position)
    print(len(points))
    return result/len(points)
   
        

