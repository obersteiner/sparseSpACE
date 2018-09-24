import numpy as np
import math


class CombiScheme:
    initialized_adaptive = False
    active_index_set = set()
    old_index_set = set()
    dim = None

    @staticmethod
    def init_adaptive_combi_scheme(dim, lmax, lmin):
        assert lmax >= lmin
        CombiScheme.dim = dim
        CombiScheme.lmin = lmin
        CombiScheme.lmax = lmax
        CombiScheme.initialized_adaptive = True
        CombiScheme.active_index_set = CombiScheme.init_active_index_set(lmax, lmin)
        CombiScheme.old_index_set = CombiScheme.init_old_index_set(lmax, lmin)

    @staticmethod
    def extendable_level(levelvec):
        assert CombiScheme.initialized_adaptive
        counter = 0
        extendable_dim = 0
        for d in range(CombiScheme.dim):
            if levelvec[d] > 1:
                counter += 1
                extendable_dim = d

        return counter == 1, extendable_dim

    @staticmethod
    def is_refinable(levelvec):
        assert CombiScheme.initialized_adaptive
        return tuple(levelvec) in CombiScheme.active_index_set

    @staticmethod
    def update_adaptive_combi(levelvec):
        assert CombiScheme.initialized_adaptive
        if not CombiScheme.is_refinable(levelvec):
            return
        refined_dims = []
        # remove this levelvec from active_index_set and add to old_index_set
        CombiScheme.active_index_set.remove(tuple(levelvec))
        CombiScheme.old_index_set.add(tuple(levelvec))
        for d in range(CombiScheme.dim):
            if CombiScheme.refine_scheme(d, levelvec):
                refined_dims.append(d)
        return refined_dims

    @staticmethod
    def refine_scheme(d, levelvec):
        assert CombiScheme.initialized_adaptive
        # print(CombiScheme.old_index_set, CombiScheme.active_index_set, levelvec, CombiScheme.lmin)
        levelvec = list(levelvec)
        levelvec[d] += 1
        for dim in range(CombiScheme.dim):
            levelvec_copy = list(levelvec)
            levelvec_copy[dim] = levelvec[dim] - 1
            if tuple(levelvec_copy) not in CombiScheme.old_index_set and not levelvec_copy[dim] < CombiScheme.lmin:
                return False
        CombiScheme.active_index_set.add(tuple(levelvec))
        return True

    @staticmethod
    def init_active_index_set(lmax, lmin):
        assert CombiScheme.initialized_adaptive
        grids = CombiScheme.getGrids(CombiScheme.dim, lmax - lmin + 1)
        grids = [tuple([l + (lmin - 1) for l in g]) for g in grids]
        print(grids)
        return set(grids)

    @staticmethod
    def init_old_index_set(lmax, lmin):
        grid_array = []
        for q in range(1, min(CombiScheme.dim, lmax - lmin + 1)):
            grids = CombiScheme.getGrids(CombiScheme.dim, lmax - lmin + 1 - q)
            grids = [tuple([l + (lmin - 1) for l in g]) for g in grids]
            grid_array.extend(grids)
        print(grid_array)
        return set(grid_array)

    @staticmethod
    def get_index_set():
        return CombiScheme.old_index_set | CombiScheme.active_index_set

    @staticmethod
    def getCombiScheme(lmin, lmax, dim, do_print=True):
        grid_array = []
        if not CombiScheme.initialized_adaptive:  # use default scheme
            for q in range(min(dim, lmax-lmin+1)):
                coefficient = (-1)**q * math.factorial(dim-1)/(math.factorial(q)*math.factorial(dim-1-q))
                grids = CombiScheme.getGrids(dim, lmax - lmin + 1 - q)
                grid_array.extend([(np.array(g, dtype=int)+np.ones(dim, dtype=int)*(lmin-1), coefficient) for g in grids])
            for i in range(len(grid_array)):
                if do_print:
                    print(i, list(grid_array[i][0]), grid_array[i][1])
        else:  # use adaptive schem
            assert(False)
            grid_array = CombiScheme.get_coefficients_to_index_set(CombiScheme.active_index_set | CombiScheme.old_index_set)
            # print(grid_dict.items())
        return grid_array

    @staticmethod
    def get_coefficients_to_index_set(index_set):
        grid_array = []
        grid_dict = {}
        for grid_levelvec in index_set:
            stencils = []
            for d in range(CombiScheme.dim):
                if grid_levelvec[d] <= CombiScheme.lmin:
                    stencils.append([0])
                else:
                    stencils.append([0, -1])
            stencil_elements = list(zip(*[g.ravel() for g in np.meshgrid(*stencils)]))
            for s in stencil_elements:
                levelvec = tuple(map(lambda x, y: x + y, grid_levelvec, s))  # adding tuples
                update_coefficient = -(abs((sum(s))) % 2) + (abs(((sum(s)) - 1)) % 2)
                if levelvec in grid_dict:
                    grid_dict[levelvec] += update_coefficient
                else:
                    grid_dict[levelvec] = update_coefficient

        for levelvec, coefficient in grid_dict.items():
            if coefficient != 0:
                grid_array.append((levelvec, coefficient))
        return grid_array

    @staticmethod
    def is_old_index(levelvec):
        return tuple(levelvec) in CombiScheme.old_index_set

    @staticmethod
    def getGrids(dim_left, values_left):
        if dim_left == 1:
            return [[values_left]]
        grids = []
        for index in range(values_left):
            levelvector = [index+1]
            grids.extend([levelvector + g for g in CombiScheme.getGrids(dim_left - 1, values_left - index)])
        return grids


    @staticmethod
    def in_index_set(levelvec):
        return tuple(levelvec) in CombiScheme.active_index_set or tuple(levelvec) in CombiScheme.old_index_set