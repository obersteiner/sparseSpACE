from StandardCombi import *
from combiScheme import *
from Grid import *


# T his class implements the standard combination technique
class DimAdaptiveCombi(StandardCombi):
    # initialization
    # a = lower bound of integral; b = upper bound of integral
    # grid = specified grid (e.g. Trapezoidal);
    def __init__(self, a, b, operation):
        self.log = logging.getLogger(__name__)
        self.dim = len(a)
        self.a = a
        self.b = b
        self.operation = operation
        self.combischeme = CombiScheme(self.dim)
        self.grid = self.operation.get_grid()
        assert (len(a) == len(b))

    # standard dimension-adaptive combination scheme for quadrature
    # lmin = minimum level; lmax = target level
    # f = function to integrate; dim=dimension of problem
    def perform_combi(self, minv, maxv, tolerance):
        start = self.a
        end = self.b
        self.operation.initialize()
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
            for i, component_grid in enumerate(self.scheme):
                if tuple(component_grid.levelvector) not in integral_dict:
                    integral = self.operation.grid.integrate(self.operation.f, component_grid.levelvector, start, end)
                    integral_dict[tuple(component_grid.levelvector)] = integral
                else:
                    integral = integral_dict[tuple(tuple(component_grid.levelvector))]
                # as error estimator we compare to the analytic solution and divide by the cost=number of points in grid
                error_array[i] = abs(integral - real_integral) / abs(real_integral) / np.prod(
                    self.operation.grid.levelToNumPoints(component_grid.levelvector)) if self.combischeme.is_refinable(component_grid.levelvector) else 0
                combiintegral += integral * component_grid.coefficient
            do_refine = True
            if max(abs(combiintegral - real_integral) / abs(real_integral)) < tolerance:
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
            self.scheme = self.combischeme.getCombiScheme(self.lmin[0], self.lmax[0], do_print=False)
            for component_grid in self.scheme:
                for d in range(self.dim):
                    self.lmax[d] = max(self.lmax[d], component_grid.levelvector[d])
        print("Final scheme:")
        self.scheme = self.combischeme.getCombiScheme(self.lmin[0], self.lmax[0], do_print=True)
        print("CombiSolution", combiintegral)
        print("Analytic Solution", real_integral)
        print("Difference", abs(combiintegral - real_integral))
        return self.scheme, abs(combiintegral - real_integral), combiintegral, errors, num_points
