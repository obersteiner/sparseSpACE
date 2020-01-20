''' better name would be ComponentGrid or Grid in general a ComponentGrid would be a child of ?'''
import numpy as np
from abc import abstractmethod

class ComponentGridInfo(object):
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
            - coefficient

    '''
    def __init__(self, levelvector, **kwargs):
        self.levelvec = levelvector
        self.dim = len(self.levelvec)
        self.coefficient = kwargs.get(coefficient, None)
        self.boundaries = kwargs.get(boundaries, list([True for __ in range(self.dim)]))
        self.N = []

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
        self.data = np.zeros(self.N)
        
    def getDim(self):
        return self.dim

    def getLevelvector(self):
        return self.levelvec

    def getNodeCount(self):
        return self.N

    def getCoefficient(self):
        assert self.coefficient != None, "No coefficient specified for this grid"
        return self.coefficient


    @abstractmethod
    def fillData(self,*args):
        pass
    
    @abstractmethod
    def getData(self):
        pass 
    
    @abstractmethod
    def interpolateData(self,level,*args,**kwargs):
        pass

class DummyGridArbitraryDim(ComponentGridInfo):
    ''' Dummy grid with boundaries, defined in unit domain of arbitrary dimension '''
    def __init__(self, levelvector, **kwargs):
        ComponentGridInfo.__init__(self,levelvector, **kwargs)
        
    def fillData(self, f=None, sdcFactor=0):
        ''' Fill data with either:
            a) values evaluated on gridpoints or
            b) specified numpy array of appropriate size 
        '''
        if f != None:
            if callable(f):
                # Fill data with values evaluated on grid points
                self.data = f(*self.coord)
                # if sdcFactor != 0:
                #     self.addSDC(sdcFactor)
            else:
                assert np.shape(f) == np.shape(self.data), "Invalid shape of provided grid data array"
                self.data = f
    
    def getData(self):
        return self.data

    def interpolateData(self, levelvector) -> np.array:
        # get level for interpolating
        new = tuple(map(float,[(2**i+1) for i in level]))
        # get factor for interpolation
        fac = np.divide(self.data, self.N)
        # interpolate grid
        return np.zoom(self.data, fac, order=1)