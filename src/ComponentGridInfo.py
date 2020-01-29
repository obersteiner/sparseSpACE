''' better name would be ComponentGrid or Grid in general a ComponentGrid would be a child of ?'''
import numpy as np
from scipy.ndimage.interpolation import zoom
from abc import abstractmethod

class ComponentGridInfo(object):
    '''
    Holds basic information of a grid used for Combination Technique
    Atributes:
        - coord: array of arrays coordinates
        - N: list with numbers of nodes in each dimension
        - data: np.array of gridpoint data

    Inputs:
        - levelvector: list of length dim determining node count in each dimension of the grid
        - **kwargs:
            - boundaries: bool list, if True, a dimension has boundary

    '''
    def __init__(self, levelvector: tuple, coefficient, boundaries=None):
        self.levelvector = levelvector
        self.dim = len(self.levelvector)
        self.coefficient = coefficient
        if boundaries == None: 
            self.boundaries = list([True for __ in range(self.dim)])
        else:
            self.boundaries=boundaries
        N = []

        coord = []
        for i in range(self.dim):
            n = 2**self.levelvector[i] + 1
            if self.boundaries[i]==True:
                coord.append(np.linspace(0,1.0,n))
                N.append(n)
            else:
                coord.append(np.linspace(0,1.0,n)[1:-1])
                N.append(n-1)

        self.coord = np.meshgrid(*coord, indexing='ij') # Corerct for 3D or 4D case ?
        self.N = tuple(N)

    def fill_data(self, f):
        ''' Fill data with either:
            a) values evaluated on gridpoints or
            b) specified numpy array of appropriate size 
        '''
        if callable(f):
            # Fill data with dummy coord values
            self.data = f(*self.coord)
        else:
            print("ComponentGrid data shape: {}".format(np.shape(f)))
            assert np.shape(f) == self.N or np.shape(f[0]) == self.N, "Invalid shape of provided grid data array"
            self.data = f
    
    def interpolate_data(self, levelvector) -> np.array:
        # get level for interpolating
        new = tuple(map(float,[(2**i+1) for i in levelvector]))
        # get factor for interpolation
        fac = np.divide(new, self.N)
        if np.shape(self.data) == self.N:
            return zoom(self.data, fac, order=1)
        else:
            interpolated_results = []
            for result in self.data:
                interpolated_results.append(zoom(result, fac, order=1))
            return np.array(interpolated_results)

    def get_dim(self):
        return self.dim

    def get_levelvector(self):
        return self.levelvector

    def get_points(self):
        return self.N

    def get_coefficient(self):
        return self.coefficient
    
    def get_data(self):
        return self.data


if __name__=="__main__":
    component_grid = ComponentGridInfo((1,2),[-1,1])
    N_x, N_y = component_grid.get_points()
    print("Levelvector: {}".format(component_grid.get_levelvector()))
    print("# of points: {}".format(component_grid.get_points()))
    print("# of points: {}, {}".format(N_x, N_y))