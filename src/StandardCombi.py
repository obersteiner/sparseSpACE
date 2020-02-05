import matplotlib.pyplot as plt
from combiScheme import *
from GridOperation import *
import importlib
import multiprocessing as mp


class StandardCombi(object):
    """This class implements the standard combination technique.

    """

    def __init__(self, a, b, operation: GridOperation, print_output=True):
        """

        :param a: Vector of lower boundaries of domain.
        :param b: Vector of upper boundaries of domain.
        :param operation: GridOperation that is used for combination.
        :param print_output: Specifies whether output should be written during combination.
        """
        self.log = logging.getLogger(__name__)
        self.dim = len(a)
        self.a = a
        self.b = b
        assert operation is not None
        self.grid = operation.get_grid()
        self.combischeme = CombiScheme(self.dim)
        self.print_output = print_output
        assert (len(a) == len(b))
        self.operation = operation
        self.do_parallel = True

    def __call__(self, interpolation_points: Sequence[Tuple[float, ...]]) -> Sequence[Sequence[float]]:
        """This method evaluates the model at the specified interpolation points using the Combination Technique.

        :param interpolation_points: List of points at which we want to evaluate/interpolate.
        :return: List of values (each a numpy array)
        """
        interpolation = np.zeros((len(interpolation_points), self.operation.point_output_length()))
        self.do_parallel = False
        if self.do_parallel:
            pool = mp.Pool(4)
            interpolation_results = pool.starmap_async(self.get_multiplied_interpolation, [(interpolation_points, component_grid) for component_grid in self.scheme]).get()
            pool.close()
            pool.join()
            for result in interpolation_results:
                interpolation += result
        else:
            for component_grid in self.scheme:
                interpolation += self.interpolate_points(interpolation_points, component_grid) * component_grid.coefficient
        return interpolation

    def interpolate_points(self, interpolation_points: Sequence[Tuple[float, ...]], component_grid: ComponentGridInfo):
        """This method evaluates the model at the specified interpolation points on the specified component grid.

        :param interpolation_points: List of points at which we want to evaluate/interpolate.
        :param component_grid: ComponentGridInfo of the specified component grid.
        :return: List of values (each a numpy array)
        """
        self.grid.setCurrentArea(self.a, self.b, component_grid.levelvector)
        return self.operation.interpolate_points(self.operation.get_component_grid_values(component_grid, self.grid.coordinate_array_with_boundary), mesh_points_grid=self.grid.coordinate_array_with_boundary,
                                                 evaluation_points=interpolation_points)

    def interpolate_grid(self, grid_coordinates: Sequence[Sequence[float]]) -> Sequence[Sequence[float]]:
        """This method evaluates the model at the specified interpolation grid using the Combination Technique.

        :param grid_coordinates: 1D grid coordinates where we want to evaluate/interpolate the model
        :return: List of values (each a numpy array)
        """
        num_points = np.prod([len(grid_d) for grid_d in grid_coordinates])
        interpolation = np.zeros((num_points, self.operation.point_output_length()))
        for component_grid in self.scheme:
            interpolation += self.interpolate_grid_component(grid_coordinates, component_grid) * component_grid.coefficient
        return interpolation

    def interpolate_grid_component(self, grid_coordinates: Sequence[Sequence[float]], component_grid: ComponentGridInfo) -> Sequence[Sequence[float]]:
        """This method evaluates the model at the specified interpolation grid on the specified component grid.

        :param grid_coordinates: 1D grid coordinates where we want to evaluate/interpolate the model
        :param component_grid: ComponentGridInfo of the specified component grid.
        :return: List of values (each a numpy array)
        """
        grid_points = list(get_cross_product(grid_coordinates))
        return self.interpolate_points(grid_points, component_grid)

    def get_multiplied_interpolation(self, interpolation_points: Sequence[Tuple[float, ...]], component_grid: ComponentGridInfo):
        """Returns the interpolation result on specified component grid at interpolation points and multiplied by  combi
        coefficient.

        :param interpolation_points: List of points at which we want to evaluate/interpolate.
        :param component_grid: ComponentGridInfo of the specified component grid.
        :return: List of values (each a numpy array)
        """
        return self.interpolate_points(interpolation_points, component_grid) * component_grid.coefficient

    def plot(self, plotdimension: int=0) -> None:
        """This method plots the model obtained by the Combination Technique.

        :param plotdimension: Dimension of the output vector that should be plotted. (0 if scalar outputs)
        :return: None
        """
        if self.dim != 2:
            print("Can only plot 2D results")
            return
        xArray = np.linspace(self.a[0], self.b[0], 10 ** 2)
        yArray = np.linspace(self.a[1], self.b[1], 10 ** 2)
        X = [x for x in xArray]
        Y = [y for y in yArray]
        points = list(get_cross_product([X, Y]))

        #f_values = np.asarray(self.interpolate_grid([X,Y]))

        X, Y = np.meshgrid(X, Y, indexing="ij")
        Z = np.zeros(np.shape(X))
        f_values = np.asarray((self(points)))
        # print(f_values)
        for i in range(len(X)):
            for j in range(len(X[i])):
                # print(X[i,j],Y[i,j],self.eval((X[i,j],Y[i,j])))
                Z[i, j] = f_values[j + i * len(X)][plotdimension]
        # Z=self.eval((X,Y))
        # print Z
        fig = plt.figure(figsize=(14, 6))

        # `ax` is a 3D-aware axis instance, because of the projection='3d' keyword argument to add_subplot
        ax = fig.add_subplot(1, 2, 1, projection='3d')

        # p = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        p = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False)
        # plt.show()
        plt.show()

    def set_combi_parameters(self, lmin: int, lmax: int) -> None:
        """Initializes the combi parameters according to minimum and maximum level.

        :param lmin: Minimum level of combination technique.
        :param lmax: Maximum level of combination technique.
        :return: None
        """
        # compute minimum and target level vector
        self.lmin = [lmin for i in range(self.dim)]
        self.lmax = [lmax for i in range(self.dim)]
        # get combi scheme
        self.scheme = self.combischeme.getCombiScheme(lmin, lmax, self.print_output)


    # lmin = minimum level; lmax = target level
    def perform_operation(self, lmin: int, lmax: int) -> Tuple[Sequence[ComponentGridInfo], float, Sequence[float]]:
        """This method performs the standard combination scheme for the chosen operation.

        :param lmin: Minimum level of combination technique.
        :param lmax: Maximum level of combination technique.
        :return: Combination scheme, error, and combination result.
        """
        assert self.operation is not None

        # initializtation
        self.set_combi_parameters(lmin, lmax)
        self.operation.initialize()

        # iterate over all component_grids and perform operation
        for component_grid in self.scheme:  # iterate over component grids
            self.operation.evaluate_levelvec(component_grid)

        # potential post processing after processing all component grids
        self.operation.post_processing()

        # get result of combination
        combi_result = self.operation.get_result()

        # obtain reference solution if available
        reference_solution = self.operation.get_reference_solution()

        # output combi_result
        if self.print_output:
            print("CombiSolution", combi_result)

        # return results
        if reference_solution is not None:
            if self.print_output:
                print("Analytic Solution", reference_solution)
                print("Difference", self.operation.get_error(combi_result, reference_solution))
            return self.scheme, self.operation.get_error(combi_result, reference_solution), combi_result
        else:
            return self.scheme, None, combi_result

    def get_num_points_component_grid(self, levelvector: Sequence[int], count_multiple_occurrences: bool):
        """This method returns the number of points contained in the specified component grid.

        :param levelvector: Level vector of the compoenent grid.
        :param count_multiple_occurrences: Indicates whether points that appear multiple times should be counted again.
        :return: Number of points in component grid.
        """
        return np.prod(self.grid.levelToNumPoints(levelvector))

    def get_total_num_points(self, doNaive: bool=False,
                             distinct_function_evals: bool=True) -> int:  # we assume here that all lmax entries are equal
        """This method calculates the total number of points used in the combination technique.

        :param doNaive: Indicates whether we should count points that appear multiple times in a grid again (False-> no)
        :param distinct_function_evals: Indicates whether we should recount points that appear in different grids.
        :return: Total number of points.
        """
        if distinct_function_evals:
            return self.operation.get_distinct_points()
        numpoints = 0
        for component_grid in self.scheme:
            pointsgrid = self.get_num_points_component_grid(component_grid.levelvector, doNaive)
            if distinct_function_evals and self.grid.isNested():
                numpoints += pointsgrid * int(component_grid.coefficient)
            else:
                numpoints += pointsgrid
        # print(numpoints)
        return numpoints

    # prints every single component grid of the combination and orders them according to levels
    def print_resulting_combi_scheme(self, filename: str=None, add_refinement: bool=True, ticks: bool=True, markersize: int=20, show_border=True, linewidth=2.0, show_levelvec=True, show_coefficient=False):
        """This method plots the the combination scheme including the points and maybe additional refinement structures.

        :param filename: If set the plot will be set to the specified filename.
        :param add_refinement: If set the refinement structure of the refinement strategy will be plotted.
        :param ticks: If set the ticks in the plots will be set.
        :param markersize: Specifies the marker size in the plot.
        :param show_border: If set the borders of the indivdual plots will be shown for each component grid.
        :param linewidth: Specifies linewidth.
        :param show_levelvec: If set level vectors will be printed above component grids.
        :param show_coefficient: If set the coefficient of component grid will be shown.
        :return: Matplotlib Figure.
        """
        fontsize = 60
        plt.rcParams.update({'font.size': fontsize})
        scheme = self.scheme
        lmin = self.lmin
        lmax = self.lmax #[self.combischeme.lmax_adaptive] * self.dim if hasattr(self.combischeme, 'lmax_adaptive') else self.lmax
        dim = self.dim
        if dim != 2:
            print("Cannot print combischeme of dimension > 2")
            return None
        fig, ax = plt.subplots(ncols=self.lmax[0] - self.lmin[0] + 1, nrows=self.lmax[1] - self.lmin[1] + 1, figsize=(10*self.lmax[0], 10*self.lmax[1]))
        # for axis in ax:
        #    spine = axis.spines.values()
        #    spine.set_visible(False)
        # get points of each component grid and plot them individually
        if lmax == lmin:
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.set_xlim([self.a[0] - 0.05, self.b[0] + 0.05])
            ax.set_ylim([self.a[1] - 0.05, self.b[1] + 0.05])
            points = self.get_points_component_grid(lmax)
            x_array = [p[0] for p in points]
            y_array = [p[1] for p in points]
            if any([math.isinf(x) for x in np.concatenate([self.a, self.b])]):
                ax.set_xlim([min(x_array) - 0.05, max(x_array) + 0.05])
                ax.set_ylim([min(y_array) - 0.05, max(y_array) + 0.05])
            else:
                ax.set_xlim([self.a[0] - 0.05, self.b[0] + 0.05])
                ax.set_ylim([self.a[1] - 0.05, self.b[1] + 0.05])
            ax.plot(x_array, y_array, 'o', markersize=markersize, color="black")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            if show_levelvec:
                ax.set_title(str(tuple(self.lmax)))
            if show_border:
                startx = self.a[0]
                starty = self.a[1]
                endx = self.b[0]
                endy = self.b[1]
                facecolor = "green"
                ax.add_patch(
                    patches.Rectangle(
                        (startx, starty),
                        endx - startx,
                        endy - starty,
                        fill=True,  # remove background,
                        alpha=0.5,
                        linewidth=linewidth, visible=True, facecolor=facecolor,edgecolor='black'
                    )
                )
            if not ticks:
                ax.axis('off')
            if add_refinement:
                self.add_refinment_to_figure_axe(ax, linewidth=linewidth)
        else:

            for i in range(lmax[0] - lmin[0] + 1):
                for j in range(lmax[1] - lmin[1] + 1):
                    ax[j, i].axis('off')

            for component_grid in scheme:
                points = self.get_points_component_grid(component_grid.levelvector)
                points_not_null = self.get_points_component_grid_not_null(component_grid.levelvector)
                x_array = [p[0] for p in points]
                y_array = [p[1] for p in points]
                x_array_not_null = [[p[0] for p in points_not_null]]
                y_array_not_null = [[p[1] for p in points_not_null]]
                grid = ax[lmax[1] - lmin[1] - (component_grid.levelvector[1] - lmin[1]), (component_grid.levelvector[0] - lmin[0])]
                grid.axis('on')
                for axdir in ("x", "y"):
                    grid.tick_params(axis=axdir, labelcolor='#345040')
                grid.xaxis.set_ticks_position('none')
                grid.yaxis.set_ticks_position('none')
                if any([math.isinf(x) for x in np.concatenate([self.a, self.b])]):
                    grid.set_xlim([min(x_array) - 0.05, max(x_array) + 0.05])
                    grid.set_ylim([min(y_array) - 0.05, max(y_array) + 0.05])
                else:
                    grid.set_xlim([self.a[0] - 0.05, self.b[0] + 0.05])
                    grid.set_ylim([self.a[1] - 0.05, self.b[1] + 0.05])
                grid.plot(x_array, y_array, 'o', markersize=markersize, color="red")
                grid.plot(x_array_not_null, y_array_not_null, 'o', markersize=markersize, color="black")
                grid.spines['top'].set_visible(False)
                grid.spines['right'].set_visible(False)
                grid.spines['bottom'].set_visible(False)
                grid.spines['left'].set_visible(False)
                if show_levelvec:
                    grid.set_title(str(tuple(component_grid.levelvector)))
                if show_border:
                    startx = self.a[0]
                    starty = self.a[1]
                    endx = self.b[0]
                    endy = self.b[1]
                    facecolor = 'limegreen' if component_grid.coefficient == 1 else 'orange'
                    grid.add_patch(
                        patches.Rectangle(
                            (startx, starty),
                            endx - startx,
                            endy - starty,
                            fill=True,  # remove background,
                            #alpha=0.5,
                            linewidth=linewidth, visible=True, facecolor=facecolor, edgecolor='black'
                        )
                    )
                if not ticks:
                    grid.axis('off')
                if add_refinement:
                    self.add_refinment_to_figure_axe(grid, linewidth=linewidth)
                if show_coefficient:
                    coefficient = str(int(component_grid.coefficient)) if component_grid.coefficient <= 0 else "+" + str(int(component_grid.coefficient))
                    grid.text(0.55, 0.55, coefficient,
                          fontsize=fontsize * 2, ha='center', color="blue")
                # for axis in ['top', 'bottom', 'left', 'right']:
                #    grid.spines[axis].set_visible(False)
        # ax1 = fig.add_subplot(111, alpha=0)
        # ax1.set_ylim([self.lmin[1] - 0.5, self.lmax[1] + 0.5])
        # ax1.set_xlim([self.lmin[0] - 0.5, self.lmax[0] + 0.5])
        #plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')

        plt.show()
        return fig

    def print_resulting_sparsegrid(self, filename: str=None, show_fig: bool=True, add_refinement: bool=True, markersize: int=30,
                                   linewidth: float=2.5, ticks: bool=True, color: str="black", show_border: bool=False):
        """This method prints the resulting sparse grid obtained by the combination technique.

        :param filename: If set the plot will be set to the specified filename.
        :param show_fig: If set the figure will be shown with plt.show().
        :param add_refinement: If set additional refinement structures are added.
        :param markersize: Specifies the marker size in plot.
        :param linewidth: Spcifies the linewidth in plot.
        :param ticks: If set ticks are shown in plot.
        :param color: Specifies the color of the points in plot.
        :param show_border: If set the borders of plot will be plotted.
        :return: Matplotlib figure.
        """
        plt.rcParams.update({'font.size': 60})
        scheme = self.scheme
        dim = self.dim
        if dim != 2 and dim != 3:
            print("Cannot print sparse grid of dimension > 3")
            return None
        if dim == 2:
            fig, ax = plt.subplots(figsize=(20, 20))
        if dim == 3:
            fig = plt.figure(figsize=(20, 20))
            ax = fig.add_subplot(111, projection='3d')

        inf_bounds = any([math.isinf(x) for x in np.concatenate([self.a, self.b])])
        if inf_bounds:
            start = None
            end = None
            for component_grid in scheme:
                points = self.get_points_component_grid(component_grid.levelvector)
                min_point = [min([point[d] for point in points]) for d in range(dim)]
                max_point = [max([point[d] for point in points]) for d in range(dim)]
                start = min_point if start is None else [min(start[d], v) for d,v in enumerate(min_point)]
                end = max_point if end is None else [max(end[d], v) for d,v in enumerate(max_point)]
            ax.set_xlim([start[0] - 0.05, end[0] + 0.05])
            ax.set_ylim([start[1] - 0.05, end[1] + 0.05])
            if dim == 3:
                ax.set_zlim([start[2] - 0.05, end[2] + 0.05])
        else:
            ax.set_xlim([self.a[0] - 0.05, self.b[0] + 0.05])
            ax.set_ylim([self.a[1] - 0.05, self.b[1] + 0.05])
            if dim == 3:
                ax.set_zlim([self.a[2] - 0.05, self.b[2] + 0.05])
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        if dim == 3:
            ax.zaxis.set_ticks_position('none')
            markersize /= 2

        # get points of each component grid and plot them in one plot
        for component_grid in scheme:
            points = self.get_points_component_grid(component_grid.levelvector)
            xArray = [p[0] for p in points]
            yArray = [p[1] for p in points]
            if dim == 2:
                plt.plot(xArray, yArray, 'o', markersize=markersize, color=color)
            if dim == 3:
                zArray = [p[2] for p in points]
                plt.plot(xArray, yArray, zArray, 'o', markersize=markersize, color=color)
            for axdir in ("x", "y"):
                ax.tick_params(axis=axdir, labelcolor='#345040')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
            if show_border:
                startx = self.a[0]
                starty = self.a[1]
                endx = self.b[0]
                endy = self.b[1]
                ax.add_patch(
                    patches.Rectangle(
                        (startx, starty),
                        endx - startx,
                        endy - starty,
                        fill=False,  # remove background,
                        alpha=1,
                        linewidth=linewidth, visible=True
                    )
                )
        if not ticks:
            ax.axis('off')
        if add_refinement and dim == 2:
            self.add_refinment_to_figure_axe(ax, linewidth=linewidth)
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
        if show_fig:
            plt.show()
        return fig

    # check if combischeme is right; assertion is thrown if not
    def check_combi_scheme(self) -> None:
        """This method performs check if the combination is valid. It counts that each point is added and subtracted so
        that contribution is 1 in the end.

        :return: None
        """
        if not self.grid.isNested():
            return
        dim = self.dim
        dictionary = {}
        for component_grid in self.scheme:
            # print ii ,component_grid
            points = self.get_points_component_grid_not_null(component_grid.levelvector)
            points = set(points)
            for p in points:
                if p in dictionary:
                    dictionary[p] += component_grid.coefficient
                else:
                    dictionary[p] = component_grid.coefficient
        # print(dictionary.items())
        for key, value in dictionary.items():
            # print(key, value)
            if value != 1:
                print(dictionary)
                print("Failed for:", key, " with value: ", value)
                for area in self.refinement.get_objects():
                    print("area dict", area.levelvec_dict)
            assert (value == 1)

    def get_points_component_grid_not_null(self, levelvec: Sequence[int]) -> Sequence[Tuple[float, ...]]:
        """This method returns the points in the component grid that are not excluded.

        :param levelvec: Level vector of component grid.
        :return: List of points.
        """
        return self.get_points_component_grid(levelvec)

    def get_points_component_grid(self, levelvec: Sequence[int]) -> Sequence[Tuple[float, ...]]:
        """This method returns the points in the component grid.

        :param levelvec: Level vector of component grid.
        :return: List of points.
        """
        self.grid.setCurrentArea(self.a, self.b, levelvec)
        points = self.grid.getPoints()
        return points

    def get_points_component_grid_1D_arrays(self, levelvec: Sequence[int]) -> Sequence[Sequence[float]]:
        """This method returns the 1D arrays of points in the component grid.

        :param levelvec: Level vector of the component grid.
        :return: List of list of points.
        """
        self.grid.setCurrentArea(self.a, self.b, levelvec)
        points = self.grid.coordinate_array
        return [points]

    def get_points_and_weights_component_grid(self, levelvec: Sequence[int]) -> Tuple[Sequence[Tuple[float, ...]], Sequence[float]]:
        """This method returns the points and the quadrature weight for specified component grid.

        :param levelvec: Level vector of component grid.
        :return: List of points and list of weights.
        """
        self.grid.setCurrentArea(self.a, self.b, levelvec)
        return self.grid.get_points_and_weights()

    def get_points_and_weights(self) -> Tuple[Sequence[Tuple[float, ...]], Sequence[float]]:
        """This method returns the points and quadrature weights of complete combination technique.

        :return: List of points and list of weights.
        """
        total_points = []
        total_weights = []
        for component_grid in self.scheme:
            points, weights = self.get_points_and_weights_component_grid(component_grid.levelvector)
            total_points.extend(points)
            # adjust weights for combination -> multiply with combi coefficient
            weights = [w * component_grid.coefficient for w in weights]
            total_weights.extend(weights)
        return np.asarray(total_points), np.asarray(total_weights)

    def get_surplusses(self) -> Sequence[Sequence[float]]:
        """This method returns all surplusses that are stored in the Grid.

        :return: Numpy array of all surplusses
        """
        surplus_op = getattr(self.grid, "get_surplusses", None)
        if callable(surplus_op):
            total_surplusses = []
            for component_grid in self.scheme:
                surplusses = self.grid.get_surplusses(component_grid.levelvector)
                total_surplusses.extend(surplusses)
            return np.asarray(total_surplusses)
        else:
            print("Grid does not support surplusses")
            return None

    def add_refinment_to_figure_axe(self, ax, linewidth: int=1) -> None:
        """This method is used to add additional refinement info to the specified axe in the matplotlib figure.

        :param ax: Axe of a matplotlib figure.
        :param linewidth: Specifies linewidth.
        :return: None
        """
        pass

    @staticmethod
    def restore_from_file(filename: str) -> 'StandardCombi':
        """This method can be used to load a StandardCombi object (or a child class) from a file.

        :param filename: Specifies filename of combi object.
        :return: StandardCombi object.
        """
        spam_spec = importlib.util.find_spec("dill")
        found = spam_spec is not None
        if found:
            import dill
            with open(filename, 'rb') as f:
                return dill.load(f)
        else:
            print("Dill library not found! Please install dill using pip3 install dill.")

    def save_to_file(self, filename: str) -> None:
        """This method can be used to store a StandardCombi object (or child class) in a file.
        
        :param filename: Specifies filename where to store combi object.
        :return: None
        """
        spam_spec = importlib.util.find_spec("dill")
        found = spam_spec is not None
        if found:
            import dill
            with open(filename, 'wb') as f:
                dill.dump(self, f)
        else:
            print("Dill library not found! Please install dill using pip3 install dill.")
