import numpy as np
import random

#this method defines a basic monte carlo integration by evaluation random points
def montecarlo(f,N,dim, a, b):
    position = np.zeros(dim)
    result = 0
    for n in range(N):
        for d in range(dim):
            position[d] = random.random()*(b[d] - a[d]) + a[d]
        result +=f.eval(position)
    return result/N
