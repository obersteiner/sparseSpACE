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
        for i in range(self.dim):
            n = 2**self.levelvector[i] + 1
            if self.boundaries[i]==True:
                N.append(n)
            else:
                N.append(n-1)                
        self.N = tuple(N)

    def fill_data(self, data):
        ''' Fill data with specified numpy array of appropriate size '''
        # print("ComponentGrid data shape: {}".format(np.shape(f)))
        assert np.shape(data)[::-1] == self.N or np.shape(data[0])[::-1] == self.N, "Invalid shape of provided grid data array"
        self.data = data
    
    def interpolate_data(self, levelvector) -> np.array:
        # get level for interpolating
        new = tuple(map(float,[(2**i+1) for i in levelvector]))
        # get factor for interpolation
        fac = np.divide(new, self.N[::-1])
        if np.shape(self.data)[::-1] == self.N:
            return zoom(self.data, fac, order=1)
        # else: # case for instationary PDEs
        #     interpolated_results = []
        #     for result in self.data:
        #         interpolated_results.append(zoom(result, fac, order=1))
        #     return np.array(interpolated_results)

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