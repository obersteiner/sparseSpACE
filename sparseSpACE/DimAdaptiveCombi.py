from sparseSpACE.StandardCombi import *
from sparseSpACE.combiScheme import *
from sparseSpACE.Grid import *


# T his class implements the standard combination technique
class DimAdaptiveCombi(StandardCombi):
    # initialization
    # a = lower bound of integral; b = upper bound of integral
    # grid = specified grid (e.g. Trapezoidal);
    def __init__(self, a, b, operation, norm=2, compute_no_cost: bool=False):
        self.log = logging.getLogger(__name__)
        self.dim = len(a)
        self.a = a
        self.b = b
        self.operation = operation
        self.combischeme = CombiScheme(self.dim)
        self.grid = self.operation.get_grid()
        self.norm = norm
        self.compute_no_cost = compute_no_cost
        assert (len(a) == len(b))

    # standard dimension-adaptive combination scheme for quadrature
    # lmin = minimum level; lmax = target level
    # f = function to integrate; dim=dimension of problem
    def perform_combi(self, minv, maxv, tolerance, max_number_of_points: int=None):
        start = self.a
        end = self.b
        self.operation.initialize()
        assert maxv == 2
        # compute minimum and target level vector
        self.lmin = [minv for i in range(self.dim)]
        self.lmax = [maxv for i in range(self.dim)]
        real_integral = self.operation.get_reference_solution()
        assert(real_integral is not None)
        self.combischeme.init_adaptive_combi_scheme(maxv, minv)
        combiintegral = 0
        self.scheme = self.combischeme.getCombiScheme(self.lmin[0], self.lmax[0])
        integral_dict = {}
        errors = []  # tracks the error evolution during the refinement procedure
        num_points = []  # tracks the number of points during the refinement procedure
        while True:
            combiintegral = 0
            self.scheme = self.combischeme.getCombiScheme(self.lmin[0], self.lmax[0], do_print=False)
            error_array = np.zeros(len(self.scheme))
            # calculate integral for function self.operation.f
            for i, component_grid in enumerate(self.scheme):
                if tuple(component_grid.levelvector) not in integral_dict:
                    integral = self.operation.grid.integrate(self.operation.f, component_grid.levelvector, start, end)
                    integral_dict[tuple(component_grid.levelvector)] = integral
                else:
                    integral = integral_dict[tuple(component_grid.levelvector)]
                combiintegral += integral * component_grid.coefficient
            # calculate errors
            for i, component_grid in enumerate(self.scheme):
                if self.combischeme.is_refinable(component_grid.levelvector):
                    # as error estimator we use the error calculation from Hemcker and Griebel
                    error_array[i] = self.calculate_surplus(component_grid, integral_dict) if self.combischeme.is_refinable(component_grid.levelvector) else 0
                    #error_array[i] = abs(integral - real_integral) / abs(real_integral) / np.prod(
                    #   self.operation.grid.levelToNumPoints(component_grid.levelvector)) if self.combischeme.is_refinable(component_grid.levelvector) else 0
            do_refine = True
            max_points_reached = False if max_number_of_points is None else self.get_total_num_points() > max_number_of_points
            if max(abs(combiintegral - real_integral) / abs(real_integral)) < tolerance or max_points_reached:
                break
            print("Current combi integral:", combiintegral)
            print("Current relative error:", max(abs(combiintegral - real_integral) / abs(real_integral)))
            errors.append(max(abs(combiintegral - real_integral) / abs(real_integral)))
            num_points.append(self.get_total_num_points(distinct_function_evals=True))
            while do_refine:
                grid_id = np.argmax(error_array)
                # print(error_array)
                print("Current error:", abs(combiintegral - real_integral) / abs(real_integral))
                print("Refining", self.scheme[grid_id].levelvector)
                refined_dims = self.combischeme.update_adaptive_combi(self.scheme[grid_id].levelvector)
                do_refine = refined_dims == []
                error_array[grid_id] = 0.0
            self.scheme = self.combischeme.getCombiScheme(self.lmin[0], self.lmax[0], do_print=True)
            for component_grid in self.scheme:
                for d in range(self.dim):
                    self.lmax[d] = max(self.lmax[d], component_grid.levelvector[d])
        print("Final scheme:")
        self.scheme = self.combischeme.getCombiScheme(self.lmin[0], self.lmax[0], do_print=True)
        print("CombiSolution", combiintegral)
        print("Analytic Solution", real_integral)
        print("Difference", abs(combiintegral - real_integral))
        return self.scheme, abs(combiintegral - real_integral), combiintegral, errors, num_points

    def calculate_surplus(self, component_grid, integral_dict):
        assert self.combischeme.is_refinable(component_grid.levelvector)
        stencils = []
        cost = 1
        for d in range(self.dim):
            if component_grid.levelvector[d] > self.lmin[d]:
                stencils.append([-1,0])
            else:
                stencils.append([0])
            cost *= 2**component_grid.levelvector[d] - 1 +  2 * int(self.grid.boundary)
        if self.compute_no_cost:
            cost = 1
        stencil_cross_product = get_cross_product(stencils)
        surplus = 0.0
        for stencil in stencil_cross_product:
            levelvector = np.array(component_grid.levelvector) + np.array(stencil)
            print(levelvector)
            integral = integral_dict[tuple(levelvector)]
            surplus += (-1)**sum(abs(np.array(stencil))) * integral
        error = LA.norm(surplus/cost,self.norm)
        print(error)
        return error
