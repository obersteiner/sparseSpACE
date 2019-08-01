import numpy as np
from typing import List, Set, Dict, Tuple, Optional, Union, Sequence


def get_cross_product(one_d_arrays: Sequence[Sequence[Union[float, int]]]) -> List[Tuple[Union[float, int], ...]]:
    dim = len(one_d_arrays)
    return list(zip(*[g.ravel() for g in
              np.meshgrid(*[one_d_arrays[d] for d in range(dim)], indexing="ij")]))


def get_cross_product_range(one_d_arrays: Sequence[Sequence[int]]) -> List[Tuple[int, ...]]:
    return get_cross_product([range(one_d_array) for one_d_array in one_d_arrays])

