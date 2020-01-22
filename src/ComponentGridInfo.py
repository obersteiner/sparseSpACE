''' better name would be ComponentGrid or Grid in general a ComponentGrid would be a child of ?'''
import numpy as np
from scipy.ndimage.interpolation import zoom
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
        - levelvector: list of length dim determining node count in each dimension of the grid
        - **kwargs:
            - boundaries: bool list, if True, a dimension has boundary

    '''
    def __init__(self, levelvector, coefficient, boundaries=None):
        self.levelvector = levelvector
        self.dim = len(self.levelvector)
        self.coefficient = coefficient
        if boundaries == None: 
            self.boundaries = list([True for __ in range(self.dim)])
        else:
            self.boundaries=boundaries
        self.N = []

        coord = []
        for i in range(self.dim):
            n = 2**self.levelvector[i] + 1
            if self.boundaries[i]==True:
                coord.append(np.linspace(0,1.0,n))
                self.N.append(n)
            else:
                coord.append(np.linspace(0,1.0,n)[1:-1])
                self.N.append(n-1)

        self.coord = np.meshgrid(*coord, indexing='ij') # Corerct for 3D or 4D case ?
        self.data = np.zeros(self.N)
        
    def fillData(self, f):
        ''' Fill data with either:
            a) values evaluated on gridpoints or
            b) specified numpy array of appropriate size 
        '''
        if callable(f):
            # Fill data with dummy coord values
            self.data = f(*self.coord)
        else:
            assert np.shape(f) == np.shape(self.data), "Invalid shape of provided grid data array"
            self.data = f
    
    def interpolateData(self, levelvector) -> np.array:
        # get level for interpolating
        new = tuple(map(float,[(2**i+1) for i in levelvector]))
        # get factor for interpolation
        fac = np.divide(new, self.N)
        # interpolate grid
        return zoom(self.data, fac, order=1)

    def getDim(self):
        return self.dim

    def getLevelvector(self):
        return self.levelvector

    def getNodeCount(self):
        return self.N

    def getCoefficient(self):
        return self.coefficient
    
    def getData(self):
        return self.data


if __name__=="__main__":
    component_grid = ComponentGridInfo((1,2),[-1,1])
    print("Levelvector: {}".format(component_grid.getLevelvector()))
    print("Node count: {}".format(component_grid.getNodeCount()))
    for item in component_grid.getNodeCount() - np.ones(2):
        print(item)