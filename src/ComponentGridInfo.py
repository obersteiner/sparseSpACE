import numpy as np
from scipy.ndimage.interpolation import zoom
from abc import abstractmethod
from scipy.interpolate import interpn

class ComponentGridInfo(object):
    """
    WARNING: a and b are not passed upon instatiation of this class yet - each component grid is always created with default values.

    Holds basic information about a component grid including the intermediate solution used for combination
    Atributes:
    - coords: array of vectors with coordinates of mesh points along each axis
    - N: tuple with numbers of nodes in each dimension (incl. boundary points) e.g. (N_x, N_y)
    - data: np.array with the component solution on this grid
    - levelvector: tuple of length dim determining node count in each dimension of the grid
    - **kwargs:
        - a, b: coordinates of diagonal points specyfying size of the domain, default: (0,..0), (1,..1) i.e unit hypercube
        - boundaries: bool list for boundaries in each dimension separately, default: with boundaries
    """
    def __init__(self, levelvector:tuple, coefficient:int, **kwargs):
        self.levelvector = tuple(levelvector)
        self.dim = len(self.levelvector)
        self.coefficient = coefficient
        self.boundaries = kwargs.get("boundaries", np.ones(self.dim, dtype=bool))
        self.a = kwargs.get("a", np.zeros(self.dim))
        self.b = kwargs.get("b",np.ones(self.dim)) 
        self.N = tuple([2**self.levelvector[i]+1 if self.boundaries[i] else 2**self.levelvector[i]-1 for i in range(self.dim)])
        coords=[]
        for i,b in enumerate(self.boundaries):
            if b:
                coords.append(np.linspace(self.a[i],self.b[i],self.N[i]))
            else:
                coords.append(np.linspace(self.a[i],self.b[i],self.N[i]+2)[1:-1])
        self.coords = coords


    def store_data(self, data):
        """ Saves the result computed on this component grid """
        self.data = data

    
    def interpolate_to_levelvec(self, levelvector: tuple, **kwargs) -> np.array:
        """ Interpolates component grid values to other grid with specified levelvector. 
            Returns data with the same structure (i.e works for instationary and vector valued PDEs as well)    
        """
        # Calculate number of points of the target grid
        boundaries = kwargs.get("boundaries", np.ones(self.dim, dtype=bool)) # assumes boundaries
        N = tuple([2**levelvector[i]+1 if boundaries[i] else 2**levelvector[i] for i in range(self.dim)])

        fac = np.divide(N, self.N)[::-1] #factor for scaling

        nests = len(np.shape(self.data))-self.dim
        if nests==0: # stationary, scalar-valued
            return zoom(self.data, fac, order=1)
        elif nests==1: # instationary, scalar / stationary, vector
            interpolated_results = []
            for result in self.data:
                interpolated_results.append(zoom(result, fac, order=1))
            return np.array(interpolated_results)
        elif nests==2: # instationary, vecor-valued
            interpolated_results = []
            for timestep in self.data:
                components = []
                for result in timestep:
                    components.append(zoom(result, fac, order=1))
                interpolated_results.append(components)
            return np.array(interpolated_results)



    def interpolate_to_points(self, evaluation_points):
        """ WARNING: Have in mind, this method works only with correct coords

            Interpolate (bilinearly) component grid values to specified evaluation_points
            Paramers:
            - evaluation_points - sequence of tuples with point coordinates [(x_1,y_1),(x_2,y_2),...]
        """    
        nests = len(np.shape(self.data))-self.dim
        if nests==0: # stationary, scalar-valued
            return interpn(self.coords[::-1], self.data, evaluation_points, method='linear')
        elif nests==1: # instationary, scalar / stationary, vector
            interpolated_results = []
            for data in self.data:
                interpn(self.coords[::-1], data, evaluation_points, method='linear')
            return np.array(interpolated_results)
        elif nests==2: # instationary, vecor-valued
            interpolated_results = []
            for timestep in self.data:
                components = []
                for result in timestep:
                    components.append(interpn(self.coords[::-1], result, evaluation_points, method='linear'))
                interpolated_results.append(components)
            return np.array(interpolated_results)


    def get_points(self):
        "Returns list of tuples of nodal points coordinates (row by row)"
        return list(zip(*[X.flatten() for X in np.meshgrid(*self.coords)]))


    def get_data_dictionary(self):
        """ Returns a dictionary of (x,y):nodal_value """
        return dict(zip(self.get_points(), self.get_data().flatten()))


    def get_dim(self):
        return self.dim

    def get_levelvector(self):
        return self.levelvector

    def get_N(self):
        return self.N

    def get_coefficient(self):
        return self.coefficient
    
    def get_data(self):
        return self.data


if __name__=="__main__":
    component_grid = ComponentGridInfo((1,2),1)
    N_x, N_y = component_grid.get_N()
    print("Boundaries: %s" %component_grid.boundaries)
    print("Levelvector: {}".format(component_grid.get_levelvector()))
    print("# of points: {}".format(component_grid.get_N()))
    print("Coords: {}".format(component_grid.coords))
    print("Points: {}".format(component_grid.get_points()))