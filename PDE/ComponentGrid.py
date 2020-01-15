'''
TO be retired? content moved to ComponentGridInfo
'''
import logging
from abc import ABCMeta, abstractmethod
import numpy as np
from numpy.core.numeric import zeros, array, log2, sqrt
from numpy.lib.index_tricks import mgrid
from scipy.ndimage.interpolation import map_coordinates, zoom
from numpy import unique
from numpy.matlib import rand
from numpy.linalg import norm
from numpy import asarray, linspace, meshgrid, add, divide
import itertools as it
import types

class ComponentGrid(object, metaclass=ABCMeta):
    '''
    Holds basic information of a grid used for Combination Technique
    like levelvector and data and provides basic methods amongst other extrapolation
    used for combining grids.
    Atributes:
        - coord: array of arrays coordinates
        - N: list with numbers of nodes in each dimension
        - data: np.array of gridpoint data

    Inputs:
        - levelvec: list of length dim determining node count in each dimension of the grid
        - **kwargs:
            - boundaries: bool list, if True, a dimension has boundary

    '''
    def __init__(self, levelvector, **kwargs):
        self.levelvec = levelvector
        self.dim = len(self.levelvec)
        if boundaries == None:
            self.boundaries = list([True for __ in range(self.dim)])
        else:
            self.boundaries = boundaries
        
        N = []
        coord = []
        for i in range(self.dim):
            n = 2**self.levelvec[i] + 1
            if self.boundaries[i]==True:
                coord.append(np.linspace(0,1.0,n))
                self.N.append(n)
            else:
                coord.append(np.linspace(0,1.0,n)[1:-1])
                self.N.append(n-1)

        self.coord = np.meshgrid(*coord, indexing='ij') # Corerct for 3D or 4D case ?
        self.N = tuple(N)
        self.data = np.zeros(self.N)
        
    def getDim(self):
        return self.dim

    def getLevelvector(self):
        return self.levelvec

    def getNodeCount(self):
        return self.N
    
    @abstractmethod
    def fillData(self,*args):
        pass
    
    @abstractmethod
    def getData(self):
        pass 
    
    @abstractmethod
    def extrapolateData(self,level,*args,**kwargs):
        pass

'''
class combiGridDummy2D(combiGrid):
    def __init__(self,levelvector,boundaries):
        combiGrid.__init__(self,levelvector,boundaries)
        self.surplus = None
        if self.boundaries!=(True,True):
            raise NotImplementedError('Dummy 2d just for grids with boundary')
        
    def fillData(self, f=None, sdcFactor=0):
        if f is None:
            self.data = np.zeros(self.N)
            s = self.data.shape
            meshgrid = mgrid[0:1:1j*s[0],0:1:1j*s[1]]
            self.data += meshgrid[0] + 1.j*meshgrid[1]
        elif isinstance(f,types.FunctionType):
            self.data = f(self.levelvec)
            if sdcFactor != 0:
               self.addSDC(sdcFactor) 
        else:
            self.data = f

    def fillSurplus(self,f):
        self.surplus = f

    def getLevelvector(self):
        return self.levelvec
        
    def getData(self):
        return self.data

    def getSurplus(self):
        return self.surplus
    
    def extrapolateData(self,level):
        buf = self.getData()
        s = buf.shape  # shape of the original array - should be the boundaries
        l = 2 ** array(level) + 1  # size of the new array  
        l=list(l)
        self.log.debug(str(s))
        self.log.debug(str(l))
        
        # #new coordinates
        coords = mgrid[0:s[0]-1:l[0]*1j,0:s[1]-1:l[1]*1j]
        
        coord = coords.reshape((2, -1))
        self.log.debug(str(coord.shape))
#         self.log.debug(str(coord))
        self.log.debug('Coordinates:')
        for i in range(2):
            self.log.debug(str(unique(coord[i])))
        
        real= map_coordinates(buf.real, coord, order=1)
        imag= map_coordinates(buf.imag, coord, order=1)

        return real.reshape(l)#+1j*imag

    def addSDC(self,factor):
        sh = self.data.size
        gamma = factor*norm(self.data.flatten(),2)*2/sqrt(sh)
        noise = gamma*rand(self.data.shape)
        self.data += noise

    def generateStrides(self,lvec):
        strides = []
        for d in range(2):
            for i in range(lvec[d]-1):
                stride = [1,1]
                stride[d] = 2**(i+1)
                strides.append(tuple(stride))
        return tuple(strides)

    def ind1(self,s):
        ind = (slice(None,None,s[0]),slice(None,None,s[1]))
        return ind
                                
    def ind2(self,s):
        ind = (slice(s[0]>>1,None,s[0]),slice(s[1]>>1,None,s[1])) 
        return ind

    def getHierSubspace(self,s,rep):
        if rep=='f':
            A = self.data
        if rep=='h':
            A = self.surplus
        old_s = log2(array(A.shape)-1)
        ind = []
        for i in range(2):
            if s[i] == 1:
                factor = int(2**(old_s[i]-s[i]))
                ind.append(slice(None,None,factor))
            else:
                factor = int(2**(old_s[i]-s[i]+1))
                ind.append(slice(factor>>1,None,factor))
        return A[tuple(ind)]

    def getNodalSubspace(self,s,rep):
        if rep=='f':
            A = self.data
        if rep=='h':
            A = self.surplus
        old_s = log2(array(A.shape)-1)
        ind = (slice(None,None,int(2**(old_s[0]-s[0]))),slice(None,None,2**(old_s[1]-s[1])))
        return A[ind]

    def hierarchizeFullGrid(self,level):
        self.surplus = array(self.data.copy())
        strides = self.generateStrides(level)
        for s in strides:
            avg_small = self.surplus[self.ind1(s)]
            zoom_factor=list(array(self.surplus.shape,dtype=float)/array(avg_small.shape))
            avg_large = zoom(avg_small.real,zoom_factor,order=1)
            surplus_level = self.surplus-avg_large
            self.surplus[self.ind2(s)] = surplus_level[self.ind2(s)]

    def dehierarchizeFullGrid(self,level):
        sh = self.data.shape
        self.data = array(self.surplus.copy())
        strides = self.generateStrides(level)
        # Dehierarchize
        for s in reversed(strides):
            avg_small = self.data[self.ind1(s)]
            zoom_factor=list(array(sh,dtype=float)/array(avg_small.shape))
            avg_large = zoom(avg_small,zoom_factor,order=1)
            self.data[self.ind2(s)] += avg_large[self.ind2(s)]
'''


class DummyGridArbitraryDim(ComponentGrid, metaclass=ABCMeta):
    ''' Dummy grid with boundaries in unit domain of arbitrary dimension'''
    def __init__(self, levelvector, **kwargs):
        ComponentGrid.__init__(self,levelvector, **kwargs)
        
    def fillData(self, f=None, sdcFactor=0):
        ''' Fill data with:
            a) values evaluated on gridpoints or
            b) specified numpy array of appropriate size 
        '''
        if f != None:
            if callable(f):
                # Fill data with values evaluated on grid points
                self.data = f(*self.coord)
                if sdcFactor != 0:
                    self.addSDC(sdcFactor)
            else:
                assert np.shape(f), "Invalid shape of provided grid data array"
                self.data = f
    
    def getData(self):
        return self.data

    def extrapolateData(self, levelvector):
        # get level for interpolating
        new = tuple(map(float,[(2**i+1) for i in level]))
        # get factor for interpolation
        fac = divide(self.data, self.N)
        # interpolate grid
        return zoom(self.data, fac,order=1)

