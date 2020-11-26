# Python modules
import time
from sparseSpACE.RefinementContainer import *
from sparseSpACE.RefinementObject import *
from sparseSpACE.ErrorCalculator import *
from sparseSpACE.Function import *
from sparseSpACE.StandardCombi import *
from sparseSpACE.GridOperation import GridOperation
from sparseSpACE.Utils import *
import sys
if sys.version_info[0] == 3 and sys.version_info[1] >= 7:
    timing = time.time_ns
else:
    timing = time.time


# This class defines the general interface and functionalties of all spatially adaptive refinement strategies
class SpatiallyAdaptivBase(StandardCombi):
    def __init__(self, a: Sequence[float], b: Sequence[float], operation: GridOperation, norm: int = np.inf,
                 timings=None, log_level: int = log_levels.WARNING, print_level: int = print_levels.NONE,
                 filename_contour_plot: str = None, filename_refinement_graph: str = None, filename_combi_scheme_plot: str = None,
                 filename_sparse_grid_plot: str = None):
        assert operation is not None
        self.log = logging.getLogger(__name__)
        self.dim = len(a)
        self.a = a
        self.b = b
        self.grid = operation.get_grid()
        self.refinements_for_recalculate = 100
        self.operation = operation
        self.norm = norm
        self.margin = 0.9
        self.calculated_solution = None
        assert (len(a) == len(b))
        self.timings = timings
        self.log_util = LogUtility(log_level=log_level, print_level=print_level)
        self.log_util.set_print_prefix('SpatiallyAdaptivBase')
        self.log_util.set_log_prefix('SpatiallyAdaptivBase')
        self.filename_contour_plot = filename_contour_plot
        self.filename_refinement_graph = filename_refinement_graph
        self.filename_combi_scheme_plot = filename_combi_scheme_plot
        self.filename_sparse_grid_plot = filename_sparse_grid_plot

    def get_num_points_component_grid(self, levelvec: Sequence[int], count_multiple_occurrences: bool) -> int:
        array2 = self.get_points_component_grid(levelvec)
        if count_multiple_occurrences:
            array2new = array2
        else:  # remove points that appear in the list multiple times
            array2new = list(set(array2))
        # print(len(array2new))
        return len(array2new)

    def evaluate_final_combi(self) -> Tuple[Sequence[float], int]:
        """Evaluates the combination from scratch using the current refinement structures.

        :return: Combisulation and number of evaluations/points
        """
        areas = self.get_areas()
        evaluation_array = np.zeros(len(areas), dtype=int)
        self.compute_solutions(areas, evaluation_array)
        num_evaluations = np.sum(evaluation_array)
        combi_solution = self.operation.get_result()
        return combi_solution, num_evaluations

    def init_adaptive_combi(self, lmin: int, lmax: int, refinement_container: RefinementContainer, tol: float) -> None:
        """This method initializes the basic parameteres of the adaptive refinement

        :param lmin: minimum level of combination for truncated combination technique (equal for all dimensions)
        :param lmax: maximum target level (equal for all dimensions)
        :param refinement_container: refinement container object to store refinement data
        :param tol: tolerance for refinement
        :return: None
        """
        assert np.isscalar(lmin)
        assert np.isscalar(lmax)
        self.tolerance = tol

        if self.reference_solution is not None:
            self.log_util.log_debug("Reference solution: {0}".format(self.reference_solution))
        else:
            self.log_util.log_debug("No reference solution present. Working purely on surplus error estimates.")
        if refinement_container is None:  # initialize refinement
            self.lmin = [lmin for i in range(self.dim)]
            self.lmax = [lmax for i in range(self.dim)]
            # calculate the combination scheme
            self.combischeme = CombiScheme(self.dim)
            self.scheme = self.combischeme.getCombiScheme(self.lmin[0], self.lmax[0], do_print=self.print_output)
            self.initialize_refinement()
            self.operation.initialize()
        else:  # use the given refinement; in this case reuse old lmin and lmax and finestWidth; works only if there was no other run in between on same object
            self.refinement = refinement_container
            self.refinement.reinit_new_objects()
        # initialize values
        self.refinements = 0
        self.counter = 1
        # self.evaluationsTotal = 0 #number of evaluations in current grid
        # self.evaluationPerArea = [] #number of evaluations per area

    def evaluate_operation(self) -> Tuple[float, float]:
        """This method evaluates the gridoperation on all component grids including initialization and finalization

        :return: global error estimate and total error (sum of all individual errors in refinement container)
        """
        # get tuples of all the combinations of refinement to access each subarea (this is the same for each component grid)
        areas = self.get_new_areas()
        evaluation_array = np.zeros(len(areas), dtype=int)
        self.init_evaluation_operation(areas)
        self.log_util.time_func("spatialAdaptBase: compute_solutions time taken ", self.compute_solutions, areas, evaluation_array)
        self.finalize_evaluation_operation(areas, evaluation_array)

        # getArea with maximal error
        self.benefit_max = self.refinement.get_max_benefit()
        self.total_error = self.refinement.get_total_error()
        self.log_util.log_debug("max surplus error: {0} total surplus error: {1}".format(self.benefit_max, self.total_error))
        self.operation.print_evaluation_output(self.refinement)
        global_error_estimate = self.operation.get_global_error_estimate(self.refinement, self.norm)
        if global_error_estimate is not None:
            return global_error_estimate, self.total_error
        else:
            return self.total_error, self.total_error

    def init_evaluation_operation(self, areas) -> None:
        """This method performs initializations which are necessary before the actual computation of the operation

        :param areas: The list of all subareas in the refinement (can be RefinementContainer if only one subares)
        :return: None
        """
        for area in areas:
            self.operation.area_preprocessing(area)

    def compute_solutions(self, areas, evaluation_array: Sequence[int]) -> None:
        """This method computes the gridoperation on all component grids

        :param areas: The list of all subareas in the refinement (can be RefinementContainer if only one subares)
        :param evaluation_array: Numpy array in which the number of evaluations per area are stored
        :return: None
        """
        # calculate operation
        for component_grid in self.scheme:  # iterate over component grids
            if self.operation.is_area_operation():
                for k, area in enumerate(areas):
                    evaluations = self.evaluate_operation_area(component_grid, area)
                    if self.grid.isNested() and self.operation.count_unique_points():
                        evaluations *= component_grid.coefficient
                    evaluation_array[k] += evaluations
            else:
                assert (False)  # not implemented yet
                points = self.get_points_component_grid(component_grid.levelvector)
                self.operation.perform_operation(points)
                self.compute_evaluations(evaluation_array, points)

    def evaluate_operation_area(self, component_grid: ComponentGridInfo, area, additional_info=None) -> int:
        """Computes the GridOperation on a subarea of the domain

        :param component_grid: ComponentGridInfo that defines the component grid
        :param area: Definition of the subarea. Usually a RefinementObject or the complete RefinementContainer
        :param additional_info: Additional info that might be passed to the operation
        :return: number of evaluations performed on subarea
        """
        modified_levelvec, do_compute = self.coarsen_grid(component_grid.levelvector, area)
        if do_compute:
            evaluations = self.operation.evaluate_area(area, modified_levelvec, component_grid, self.refinement,
                                                       additional_info)
            return evaluations
        else:
            return 0

    def refine(self) -> None:
        """This method performs one refinement step, which might refine multiple RefinementObjects

        :return: None
        """
        # split all cells that have an error close to the max error
        self.prepare_refinement()
        self.refinement.clear_new_objects()
        margin = self.margin
        quit_refinement = False
        num_refinements = 0
        while True:  # refine all areas for which error is within margin
            # get next area that should be refined
            found_object, position, refine_object = self.refinement.get_next_object_for_refinement(
                tolerance=self.benefit_max * margin)
            if found_object and not quit_refinement:  # new area found for refinement
                self.refinements += 1
                num_refinements += 1
                # print("Refining position", position)
                quit_refinement = self.do_refinement(refine_object, position)

            else:  # all refinements done for this iteration -> reevaluate operation and check if further refinements necessary
                self.log_util.log_debug("Finished refinement")
                self.log_util.log_debug("Refined {0} times".format(num_refinements))
                self.refinement_postprocessing()
                break

        if self.recalculate_frequently and self.refinements / self.refinements_for_recalculate > self.counter:
            self.refinement.reinit_new_objects()
            self.evaluationPerArea = []
            self.evaluationsTotal = 0
            self.counter += 1
            self.log_util.log_debug("recalculating errors")

    def performSpatiallyAdaptiv(self, lmin: int=1, lmax: int=2, errorOperator: ErrorCalculator=None, tol: float= 10 ** -2,
                                refinement_container: RefinementContainer=None, do_plot: bool=False, recalculate_frequently: bool=False, test_scheme: bool=False,
                                reevaluate_at_end: bool=False, max_time: float=None, max_evaluations: int=None,
                                print_output: bool=True, min_evaluations: int=1, solutions_storage: dict=None, evaluation_points=None, single_step: bool=False
                                ) -> Tuple[RefinementContainer, Sequence[ComponentGridInfo], Sequence[int], Sequence[float], Sequence[float], Sequence[int], Sequence[float]]:
        """This is the main method for the spatially adaptive refinement strategy

        :param lmin: Minimum level for truncated combination technique (equal for all dimensions)
        :param lmax: Maximum level for combination technique (referred to as target level)
        :param errorOperator: ErrorCalculator object that calculates the errors within the RefinementContainer
        :param tol: Tolerance at which refinement is stopped
        :param refinement_container: Refinement from old refinement which should be continued
        :param do_plot: Boolean to indicate whether plots should be created during refinement
        :param recalculate_frequently: Boolean to indicate whether we should frequently restart computation from scratch
                                       This can be helpful to avoid the accumulation of rounding errors.
        :param test_scheme: Test the validity of the combination scheme at the end of refinement (for debugging)
        :param reevaluate_at_end: Boolean to indicate if we should reevaluate the whole combination after refinement
        :param max_time: Maximum compute time. The refinement will stop when it exceeds this time.
        :param max_evaluations: Maximum number of points. The refinement will stop when it exceeds this limit.
        :param print_output: Indicates whether output should be written during combination.
        :param min_evaluations: Minimum number of points. The refinement will not stop until it exceeds this limit.
        :param solutions_storage: #toDo
        :param evaluation_points: Number of points at which we want to interpolate the approximated model. This will
                                  generate the values at the points for each refinement step to analyze convergence.
        :param single_step: perform only a single refinement regardless of other parameters
        :return: #toDo

        Parameters
        ----------
        """
        assert self.operation is not None
        self.errorEstimator = errorOperator
        self.recalculate_frequently = recalculate_frequently
        self.print_output = print_output
        self.reference_solution = self.operation.get_reference_solution()
        self.init_adaptive_combi(lmin, lmax, refinement_container, tol)
        self.error_array = []
        self.surplus_error_array = []
        self.interpolation_error_arrayL2 = []
        self.interpolation_error_arrayMax = []
        self.num_point_array = []
        self.test_scheme = test_scheme
        self.reevaluate_at_end = reevaluate_at_end
        self.do_plot = do_plot
        self.calculated_solution = None
        self.solutions_storage = solutions_storage
        self.evaluation_points = evaluation_points
        self.single_step = single_step
        # for compatibility with old code


        return self.continue_adaptive_refinement(tol=tol, max_time=max_time, max_evaluations=max_evaluations, min_evaluations=min_evaluations)

    def continue_adaptive_refinement(self, tol: float=10 ** -3, max_time: float=None, max_evaluations: int=None, min_evaluations: int=1) -> Tuple[RefinementContainer, Sequence[ComponentGridInfo], Sequence[int], Sequence[float], Sequence[float], Sequence[int], Sequence[float]]:
        """Continues the adaptive refinement with potentially new limits.

        :param tol: Tolerance at which refinement is stopped
        :param max_time: Maximum compute time. The refinement will stop when it exceeds this time.
        :param max_evaluations: Maximum number of points. The refinement will stop when it exceeds this limit.
        :param min_evaluations: Minimum number of points. The refinement will not stop until it exceeds this limit.
        :return:
        """
        start_time = time.perf_counter()
        while True:
            if self.single_step:
                max_evaluations = self.last_point_count + 1 if self.last_point_count is not None else max_evaluations
            error, surplus_error = self.log_util.time_func("spatialAdaptBase: evaluate_operation time taken ", self.evaluate_operation)
            self.initialize_grid()
            self.error_array.append(error)
            self.surplus_error_array.append(surplus_error)
            self.num_point_array.append(self.get_total_num_points(distinct_function_evals=True))
            if self.evaluation_points is not None:
                interpolated_values = np.asarray(self.__call__(self.evaluation_points))
                real_values = np.asarray([self.operation.eval_analytic(point) for point in self.evaluation_points])
                diff = [real_values[i]-interpolated_values[i] for i in range(len(self.evaluation_points))]
                #print(interpolated_values, diff)
                self.interpolation_error_arrayL2.append(scipy.linalg.norm(diff, 2))
                self.interpolation_error_arrayMax.append(scipy.linalg.norm(diff, np.inf))

            if self.print_output:
                self.log_util.log_debug("Current error: {0}".format(error))

            def find_unique_filename(x: str = None):
                if x:
                    import os
                    while os.path.isfile(x+'.png'):
                        x = x + '+'
                    return x
                else:
                    return None

            if self.do_plot:
                print("Contour plot:")
                self.plot(filename=find_unique_filename(self.filename_contour_plot), contour=True)
            num_evaluations = self.get_total_num_points()
            if self.solutions_storage is not None:
                assert not self.reevaluate_at_end, "Solutions are only available in the end"
                # Remember the solutions for each number of evaluations
                self.solutions_storage[num_evaluations] = self.operation.get_result()
            # Check if conditions are met to abort refining
            if error <= tol and num_evaluations >= min_evaluations:
                break
            if max_evaluations is not None and num_evaluations > max_evaluations:
                break
            if max_time is not None and time.time() - start_time > max_time:
                break
            if self.single_step:
                self.last_point_count = self.get_total_num_points()
            # refine further
            self.log_util.time_func("spatialAdaptBase: refine time taken", self.refine)
            if self.do_plot:
                import os
                print("Refinement Graph:")
                self.draw_refinement(filename=find_unique_filename(self.filename_refinement_graph))
                print("Combi Scheme:")
                self.print_resulting_combi_scheme(filename=find_unique_filename(self.filename_combi_scheme_plot), markersize=5)
                print("Resulting Sparse Grid:")
                self.print_resulting_sparsegrid(filename=find_unique_filename(self.filename_sparse_grid_plot), markersize=10)
        # finished adaptive algorithm
        #if self.print_output:
        self.log_util.log_info("Number of refinements {0}".format(self.refinements))
        self.log_util.log_info("Number of distinct points used during the refinement: {0}".format(self.get_total_num_points()))
        self.log_util.log_info("Time used adaptive (s): {0}".format(time.perf_counter() - start_time))
        self.log_util.log_info("Final error: {0}".format(error))
        if self.test_scheme:
            self.check_combi_scheme()
        if self.reevaluate_at_end:
            # evaluate operation again from scratch
            combi_result, number_of_evaluations = self.evaluate_final_combi()
        else:
            combi_result = self.operation.get_result()
            number_of_evaluations = self.refinement.evaluationstotal
        #self.operation.set_function(None)
        self.calculated_solution = combi_result
        return self.refinement, self.scheme, self.lmax, combi_result, number_of_evaluations, self.error_array, self.num_point_array, self.surplus_error_array, self.interpolation_error_arrayL2, self.interpolation_error_arrayMax

    @abc.abstractmethod
    def initialize_grid(self):
        """
        This method initializes the grid (e.g. useful for initialize caching inside of a grid)

        :return: None
        """
        pass

    @abc.abstractmethod
    def initialize_refinement(self):
        """This method initializes the refinement container. This is specific to the indivudal strategy.

        :return: None
        """
        pass

    @abc.abstractmethod
    def get_points_component_grid(self, levelvec: Sequence[int]) -> Sequence[Tuple[float, ...]]:
        """This method returns the points that are contained in the component grid.

        :param levelvec: Level vector of the componenet grid
        :return: List of all points.
        """
        return

    @abc.abstractmethod
    def do_refinement(self, area, position) -> bool:
        """This method refines a specific area which is located at the specified position in the container.

        :param area: Area to refine. This is only relevant to ExtendSplit and Cell scheme.
        :param position: Position of element in refinement container. Used in SingleDimension method.
        :return: Boolean that indicates whether refinement should be stopped after this.
        """
        pass

    def prepare_refinement(self) -> None:
        """Method that initializes necessary data at beginning of refinement procedure. Can be overwritten if needed.

        :return: None
        """
        pass

    def refinement_postprocessing(self) -> None:
        """This method performs postprocessing steps that are performed at the end of refinement procedure.

        :return: None
        """
        removed_objects = self.refinement.apply_remove()
        self.operation.process_removed_objects(removed_objects)
        self.refinement.refinement_postprocessing()

    def calc_error(self, objectID) -> None:
        """This method calculates the error of the specified object in the refinement container.

        :param objectID: Position of the object in the RefinementContainer. This is an int or Tuple[Int,...] (MetaCont.)
        :return: None
        """
        self.refinement.calc_error(objectID, self.norm)

    def get_new_areas(self):
        """This method returns all the areas that were created during refinement -> used for computing only new delta

        :return: List of areas. Might contain only one element if no subareas exist.
        """
        return self.refinement.get_new_objects()

    def get_areas(self):
        """This method returns all the areas that are contained in RefinementContainer.

        :return: List of areas. Might contain only one element if no subareas exist.
        """
        return self.refinement.get_objects()

    def draw_refinement(self, filename: str=None, markersize: int=10):
        """This method plots the refinement structures of the method. Can be implemented by the indivudal strategy.

        :param filename:
        :param markersize:
        :return:
        """
        pass

    def coarsen_grid(self, levelvector: Sequence[int], area):
        """This method can be used to modify the levelvector according to the refinement scheme. Overwrite if needed.

        :param levelvector: Level vector of component grid.
        :param area: Subarea of the domain that we are currently computing for.
        :return: Modified level vector and boolean that indicates if we should compute this area at all.
        """
        return levelvector, True

    def finalize_evaluation_operation(self, areas, evaluation_array: Sequence[int]) -> None:
        """This method finalizes the computation of the GridOperation after all component grids have been processed.

        :param areas: List of subareas
        :param evaluation_array: List with number of evaluations/points for each subarea.
        :return: None
        """
        assert len(areas) == len(evaluation_array)

        self.log_util.log_debug("Curent number of function evaluations {0}".format(self.get_total_num_points()))

        for area in areas:
            self.operation.area_postprocessing(area)

        self.operation.post_processing()

        for k in range(len(areas)):
            i = k + self.refinement.size() - self.refinement.new_objects_size()
            self.refinement.set_evaluations(i, evaluation_array[k])

        for k in range(len(areas)):
            i = k + self.refinement.size() - self.refinement.new_objects_size()
            self.calc_error(i)
            self.refinement.set_benefit(i)

    def has_basis_grid(self):
        """This method indicates whether the grid defines basis functions.

        :return: Boolean if has basis function or not.
        """
        return isinstance(self.grid, GlobalBasisGrid)
