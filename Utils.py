import numpy as np

def get_cross_product(one_d_arrays):
    dim = len(one_d_arrays)
    return list(
        zip(*[g.ravel() for g in
              np.meshgrid(*[one_d_arrays[d] for d in range(dim)], indexing="ij")]))

def get_cross_product_range(one_d_arrays):
    return get_cross_product([range(one_d_array) for one_d_array in one_d_arrays])

