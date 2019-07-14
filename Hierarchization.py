import numpy as np

class HierarchizationLSG(object):
    def __init__(self, grid):
        self.grid = grid

    def __call__(self, f, numPoints, grid):
        self.grid = grid
        self.dim = len(numPoints)
        grid_values = np.empty((f.output_length(), np.prod(numPoints)))
        #print(numPoints)
        #print(self.grid.get_coordinates())
        for d in range(self.dim):
            assert numPoints[d] == len(self.grid.get_coordinates_dim(d))
        for d in range(self.dim):
            grid_values = self.hierarchize_poles_for_dim(grid_values, numPoints, f, d, d==0)
        return grid_values

    def hierarchize_poles_for_dim(self, grid_values, numPoints, f, d, first_dimension):
        self.dim = len(numPoints)
        numPoints_slice = np.array(numPoints)
        numPoints_slice[d] = 1
        # create all indeces in d-1 dimensional slice
        point_indeces = self.get_cross_product_range(numPoints_slice)
        # iterate over all indeces in slice (0 at dimension d)
        for point_index in point_indeces:
            # create array of function values through pole
            pole_values = np.empty((f.output_length(), numPoints[d]))
            pole_positions = []
            for i in range(numPoints[d]):
                pole_index = point_index[:d] + (i,) + point_index[d+1:]
                position = np.empty(self.dim)
                for dim in range(self.dim):
                    position[dim] = self.grid.get_coordinates_dim(dim)[pole_index[dim]]
                #print(point_index, pole_index, position, self.get_1D_coordinate(pole_index, numPoints))
                pole_positions.append(position)
                #print(pole_positions)
                if first_dimension:
                    pole_values[:,i] = f(position)
                else:
                    pole_values[:,i] = grid_values[:,self.get_1D_coordinate(pole_index, numPoints)]
            matrix = np.empty((numPoints[d], numPoints[d]))
            for i in range(numPoints[d]):
                for j in range(numPoints[d]):
                    #print(len(self.grid.grids[d].splines), numPoints[d])
                    #print(pole_positions)
                    matrix[i, j] = self.grid.get_basis(d, j)(pole_positions[i][d])
                    #print(self.grid.grids[d].splines[j].knots)
            #print(matrix)
            #print(pole_positions,d)
            for n in range(f.output_length()):
                hierarchized_values = np.linalg.solve(matrix, pole_values[n,:])
                for i in range(numPoints[d]):
                    pole_index = point_index[:d] + (i,) + point_index[d+1:]
                    grid_values[n,self.get_1D_coordinate(pole_index, numPoints)] = hierarchized_values[i]
        #print(grid_values)
        return grid_values

    def get_cross_product_range(self, one_d_arrays):
        return list(
            zip(*[g.ravel() for g in
                  np.meshgrid(*[range(one_d_arrays[d]) for d in range(self.dim)])]))

    def get_1D_coordinate(self, index_vector, numPoints):
        offsets = np.ones(self.dim, dtype=np.int64)
        for i in range(self.dim):
            if i != 0:
                offsets[i] = offsets[i - 1] * int(numPoints[i - 1])
        index = np.sum(np.array(index_vector)*offsets)
        return index
