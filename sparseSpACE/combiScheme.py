import numpy as np
import math
from sparseSpACE.ComponentGridInfo import ComponentGridInfo
from typing import List, Set, Tuple
from sparseSpACE.Utils import *

class CombiScheme:
    def __init__(self, dim: int):
        self.initialized_adaptive = False
        self.active_index_set = set()
        self.old_index_set = set()
        self.dim = dim

    # This method initializes the adaptive combination scheme. Here we create the old and the active index set
    # for the standard scheme with specified maximum and minimum level.
    def init_adaptive_combi_scheme(self, lmax: int, lmin: int) -> None:
        assert lmax >= lmin
        assert lmax >= 0
        assert lmin >= 0
        self.lmin = lmin
        self.lmax = lmax
        self.initialized_adaptive = True  # type: bool
        self.active_index_set = CombiScheme.init_active_index_set(lmax, lmin, self.dim)  # type: Set[Tuple[int, ...]]
        self.old_index_set = CombiScheme.init_old_index_set(lmax, lmin, self.dim)  # type: Set[Tuple[int, ...]]
        self.lmax_adaptive = lmax  # type: int

    # This method initializes the subspaces for a full grid. This method should only be used for plotting as it violates
    # the basic properties of the index sets for adaptation.
    def init_full_grid(self, lmax: int, lmin: int) -> None:
        assert lmax >= lmin
        assert lmax >= 0
        assert lmin >= 0
        self.lmin = lmin
        self.lmax = lmax
        self.initialized_adaptive = True  # type: bool

        self.active_index_set = set()
        self.old_index_set = CombiScheme.init_old_index_set(lmax, lmin, self.dim)  # type: Set[Tuple[int, ...]]
        for i in range(1+lmax-lmin):
            self.old_index_set = self.old_index_set | CombiScheme.init_active_index_set(lmax, lmin+i, self.dim)  # type: Set[Tuple[int, ...]]
        self.lmax_adaptive = lmax  # type: int

    def extendable_level(self, levelvec: List[int]) -> Tuple[bool, int]:
        assert self.initialized_adaptive
        counter = 0
        extendable_dim = 0
        for d in range(self.dim):
            if levelvec[d] > 1:
                counter += 1
                extendable_dim = d
        return counter == 1, extendable_dim

    # checks if the component grid is refinable, i.e. it is in the active index set
    def is_refinable(self, levelvec: List[int]) -> bool:
        assert self.initialized_adaptive
        return tuple(levelvec) in self.active_index_set

    # This method is used to refine the component grid with the specified levelvec.
    # It tries to add all forward neighbours of the grid and adds all of them that can be added.
    # This method returns the dimensions for which forward neighbours were added to the scheme. If this list is empty,
    # no grid was added.
    def update_adaptive_combi(self, levelvec: List[int]) -> List[int]:
        assert self.initialized_adaptive
        if not self.is_refinable(levelvec):
            return
        refined_dims = []
        # remove this levelvec from active_index_set and add to old_index_set
        self.active_index_set.remove(tuple(levelvec))
        self.old_index_set.add(tuple(levelvec))
        for d in range(self.dim):
            if self.__refine_scheme(d, levelvec):
                refined_dims.append(d)
        return refined_dims

    def has_forward_neighbour(self, levelvec: List[int]) -> bool:
        assert self.initialized_adaptive
        for d in range(self.dim):
            temp = list(levelvec)
            temp[d] += 1
            if tuple(temp) in self.active_index_set or tuple(temp) in self.old_index_set:
                return True
        return False

    # This method tries to add the forward neighbour in dimension d for the grid with the specified levelvector.
    # If the grid was added successfully the return value will be True, otherwise False.
    def __refine_scheme(self, d: int, levelvec: List[int]) -> Set[Tuple[int, ...]]:
        assert self.initialized_adaptive
        # print(CombiScheme.old_index_set, CombiScheme.active_index_set, levelvec, CombiScheme.lmin)
        levelvec = list(levelvec)
        levelvec[d] += 1
        for dim in range(self.dim):
            levelvec_copy = list(levelvec)
            levelvec_copy[dim] = levelvec[dim] - 1
            if tuple(levelvec_copy) not in self.old_index_set and not levelvec_copy[dim] < self.lmin:
                return False
        self.active_index_set.add(tuple(levelvec))
        self.lmax_adaptive = max(self.lmax_adaptive, levelvec[d])
        return True

    # This method initializes the active index set for the standard combination technique with specified maximum
    # and minimum level. Dim specifies the dimension of the problem.
    @staticmethod
    def init_active_index_set(lmax: int, lmin: int, dim: int) -> Set[Tuple[int, ...]]:
        grids = CombiScheme.getGrids(dim, lmax - lmin + 1)
        grids = [tuple([l + (lmin - 1) for l in g]) for g in grids]
        return set(grids)

    # This method initializes the old index set for the standard combination technique with specified maximum
    # and minimum level. Dim specifies the dimension of the problem.
    @staticmethod
    def init_old_index_set(lmax: int, lmin: int, dim: int) -> Set[Tuple[int, ...]]:
        grid_array = []
        for q in range(1, lmax - lmin + 1):
            grids = CombiScheme.getGrids(dim, lmax - lmin + 1 - q)
            grids = [tuple([l + (lmin - 1) for l in g]) for g in grids]
            grid_array.extend(grids)
        return set(grid_array)

    # This method returns the whole index set, i.e. the union of the old and active index set
    def get_index_set(self) -> Set[Tuple[int, ...]]:
        return self.old_index_set | self.active_index_set

    def get_active_indices(self) -> Set[Tuple[int, ...]]:
        return self.active_index_set

    # This method returns a list containing the whole combination scheme for the specified minimum and maximum level.
    # In case we have initialized the dimension adaptive scheme (with init_adaptive_combi_scheme) it returns the
    # current adaptive combination scheme. Here only the grids with a non-zero coeficient are returned.
    # In the adaptive case lmin and lmax parameters are not used.
    # do_print can be set to true of we want to print the combination scheme to standard output.
    def getCombiScheme(self, lmin: int=1, lmax: int=2, do_print: bool=True) -> List[ComponentGridInfo]:
        grid_array = []
        if not self.initialized_adaptive:  # use default scheme
            for q in range(min(self.dim, lmax-lmin+1)):
                coefficient = (-1)**q * math.factorial(self.dim-1)/(math.factorial(q)*math.factorial(self.dim-1-q))
                grids = CombiScheme.getGrids(self.dim, lmax - lmin + 1 - q)
                grid_array.extend([ComponentGridInfo(levelvector=np.array(g, dtype=int)+np.ones(self.dim, dtype=int)*(lmin-1), coefficient=coefficient) for g in grids])
            for i in range(len(grid_array)):
                if do_print:
                    print(i, list(grid_array[i].levelvector), grid_array[i].coefficient)
        else:  # use adaptive schem
            assert self.initialized_adaptive
            grid_array = self.get_coefficients_to_index_set(self.active_index_set | self.old_index_set)
            # print(grid_dict.items())
            for i in range(len(grid_array)):
                if do_print:
                    print(i, list(grid_array[i].levelvector), grid_array[i].coefficient)
        return grid_array

    # This method computes the coefficient for all component grids (identified by their levelvector)
    # in the specified index set. It returns a list of ComponentGridInfo Structure containing all component grids with
    # non-zero coefficients.
    def get_coefficients_to_index_set(self, index_set: Set[Tuple[int, ...]]) -> List[ComponentGridInfo]:
        grid_array = []
        grid_dict = {}
        for grid_levelvec in index_set:
            stencils = []
            for d in range(self.dim):
                if grid_levelvec[d] <= self.lmin:
                    stencils.append([0])
                else:
                    stencils.append([0, -1])
            stencil_elements = get_cross_product(stencils)
            for s in stencil_elements:
                levelvec = tuple(map(lambda x, y: x + y, grid_levelvec, s))  # adding tuples
                update_coefficient = -(abs((sum(s))) % 2) + (abs(((sum(s)) - 1)) % 2)
                if levelvec in grid_dict:
                    grid_dict[levelvec] += update_coefficient
                else:
                    grid_dict[levelvec] = update_coefficient

        for levelvec, coefficient in grid_dict.items():
            if coefficient != 0:
                grid_array.append(ComponentGridInfo(levelvector=levelvec, coefficient=coefficient))
        return grid_array

    # This method checks if the specified levelvector is contained in the old index set.
    def is_old_index(self, levelvec: List[int]) -> bool:
        return tuple(levelvec) in self.old_index_set

    # This method computes recursively all possible level vectors of dimension dim_left
    # that have an l_1 norm of values_left. This is used to efficiently compute the standard combination scheme.
    @staticmethod
    def getGrids(dim_left: int, values_left: int) -> List[List[int]]:
        if dim_left == 1:
            return [[values_left]]
        grids = []
        for index in range(values_left):
            levelvector = [index+1]
            grids.extend([levelvector + g for g in CombiScheme.getGrids(dim_left - 1, values_left - index)])
        return grids

    # This method checks if the specified levelvector is contained in the index set.
    def in_index_set(self, levelvec: List[int]) -> bool:
        return tuple(levelvec) in self.active_index_set or tuple(levelvec) in self.old_index_set
