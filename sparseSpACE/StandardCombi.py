import time
import sparseSpACE
import matplotlib.pyplot as plt
from matplotlib import cm
from sparseSpACE.combiScheme import *
from sparseSpACE.GridOperation import *
import importlib
import multiprocessing as mp
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sparseSpACE import GridOperation
from sparseSpACE.Utils import *


class StandardCombi(object):
    """This class implements the standard combination technique.

    """

    def __init__(self, a, b, operation: GridOperation, print_output: bool = False, norm: int = 2,
                 log_level: int = log_levels.WARNING, print_level: int = print_levels.NONE):
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
        self.norm = norm
        self.log_util = LogUtility(log_level=log_level, print_level=print_level)
        # for compatibility with old code
        if print_output is True and print_level == print_levels.NONE:
            self.log_util.set_print_level(print_levels.INFO)
        self.log_util.set_print_prefix('StandardCombi')
        self.log_util.set_log_prefix('StandardCombi')

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
        return self.operation.interpolate_points_component_grid(component_grid, mesh_points_grid=None,
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

    def plot(self, filename: str = None, plotdimension: int = 0, contour=False) -> None:
        """This method plots the model obtained by the Combination Technique.

        :param plotdimension: Dimension of the output vector that should be plotted. (0 if scalar outputs)
        :return: None
        """
        if self.dim != 2:
            self.log_util.log_warning("Can only plot 2D results")
            return
        fontsize = 30
        plt.rcParams.update({'font.size': fontsize})
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
        if contour:
            fig = plt.figure(figsize=(20, 10))

            # `ax` is a 3D-aware axis instance, because of the projection='3d' keyword argument to add_subplot
            ax = fig.add_subplot(1, 2, 1, projection='3d')

            # p = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

            ax = fig.add_subplot(1, 2, 2)
            # TODO why do I have to transpose here so it plots in the right orientation?
            p = ax.imshow(np.transpose(Z), extent=[0.0, 1.0, 0.0, 1.0], origin='lower', cmap=cm.coolwarm)
            # ax.axis(aspect="image")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(p, cax=cax)
        else:
            fig = plt.figure(figsize=(20, 10))

            # `ax` is a 3D-aware axis instance, because of the projection='3d' keyword argument to add_subplot
            ax = fig.add_subplot(1, 1, 1, projection='3d')

            # p = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            p = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            # TODO make colorbar look nicer
            fig.colorbar(p, ax=ax)
        # plt.show()
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
        plt.show()
        # reset fontsize to default so it does not affect other figures
        plt.rcParams.update({'font.size': plt.rcParamsDefault.get('font.size')})

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
    def perform_operation(self, lmin: int, lmax: int, plot:bool=False) -> Tuple[Sequence[ComponentGridInfo], float, Sequence[float]]:
        """This method performs the standard combination scheme for the chosen operation.

        :param lmin: Minimum level of combination technique.
        :param lmax: Maximum level of combination technique.
        :return: Combination scheme, error, and combination result.
        """
        start_time = time.perf_counter()
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
            self.log_util.log_debug("CombiSolution".format(combi_result))

        if plot:
            print("Combi scheme:")
            self.print_resulting_combi_scheme()
            print("Sparse Grid:")
            self.print_resulting_sparsegrid()
        self.log_util.log_info("Time used (s):" + str(time.perf_counter() - start_time))
        self.log_util.log_info("Number of distinct points used during the refinement (StdCombi): {0}".format(self.get_total_num_points()))
        # return results
        if reference_solution is not None:
            self.log_util.log_debug("Analytic Solution ".format(reference_solution))
            self.log_util.log_debug("Difference ".format(self.operation.compute_difference(combi_result, reference_solution, self.norm)))
            return self.scheme, self.operation.compute_difference(combi_result, reference_solution, self.norm), combi_result
        else:
            return self.scheme, None, combi_result

    def get_num_points_component_grid(self, levelvector: Sequence[int], count_multiple_occurrences: bool):
        """This method returns the number of points contained in the specified component grid.

        :param levelvector: Level vector of the compoenent grid.
        :param count_multiple_occurrences: Indicates whether points that appear multiple times should be counted again.
        :return: Number of points in component grid.
        """
        return np.prod(self.grid.levelToNumPoints(levelvector))

    def get_total_num_points(self, doNaive: bool = False,
                             distinct_function_evals: bool = True) -> int:  # we assume here that all lmax entries are equal
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
    def print_resulting_combi_scheme(self, filename: str=None, add_refinement: bool=True, ticks: bool=True, markersize: int=20, show_border: bool=True, linewidth: float=2.0, show_levelvec: bool=True, show_coefficient: bool=False, fontsize: int=40, figsize: float=10, fill_boundary_points: bool=False, consider_not_null: bool=False, operation: GridOperation=None, add_complete_full_grid_space: bool=False):
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
        plt.rcParams.update({'font.size': fontsize})
        scheme = self.scheme
        lmin = self.lmin
        lmax = self.lmax #[self.combischeme.lmax_adaptive] * self.dim if hasattr(self.combischeme, 'lmax_adaptive') else self.lmax
        dim = self.dim
        if dim != 2:
            self.log_util.log_warning("Cannot print combischeme of dimension > 2")
            return None
        ncols = self.lmax[0] - self.lmin[0] + 1
        nrows = self.lmax[1] - self.lmin[1] + 1
        fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(figsize*self.lmax[0], figsize*self.lmax[1]))
        # for axis in ax:
            # spine = axis.spines.values()
            # spine.set_visible(False)
        # for axis in fig.axes:
            # for key, value in axis.spines.items():
                # spine = value
                # spine.set_visible(False)
        # get points of each component grid and plot them individually
        if lmax == lmin:
            component_grid = scheme[0]
            self.plot_points_component_grid(component_grid, consider_not_null, fill_boundary_points, ax,
                                            linewidth, markersize, operation, show_border, ticks)
            if show_levelvec:
                ax.set_title(str(tuple(self.lmax)))
            if add_refinement:
                self.add_refinment_to_figure_axe(ax, linewidth=linewidth)
            if operation is not None:
                operation.plot_component_grid(scheme[0], ax)
        else:

            for i in range(lmax[0] - lmin[0] + 1):
                for j in range(lmax[1] - lmin[1] + 1):
                    ax[j, i].axis('off')
            if add_complete_full_grid_space:
                combischeme_full = CombiScheme(self.dim)
                combischeme_full.init_full_grid(self.lmax[0], self.lmin[0])
                component_grids_full =  []
                for levelvector in combischeme_full.get_index_set():
                    component_grids_full.append(ComponentGridInfo(levelvector, 0))
                scheme = component_grids_full + scheme
            for component_grid in scheme:
                grid = ax[lmax[1] - lmin[1] - (component_grid.levelvector[1] - lmin[1]), (component_grid.levelvector[0] - lmin[0])]
                self.plot_points_component_grid(component_grid, consider_not_null, fill_boundary_points, grid,
                                                linewidth, markersize, operation, show_border, ticks)
                if show_levelvec:
                    grid.set_title(str(tuple(component_grid.levelvector)))
                if component_grid.coefficient != 0:
                    if add_refinement:
                        self.add_refinment_to_figure_axe(grid, linewidth=linewidth)
                    if show_coefficient:
                        coefficient = str(int(component_grid.coefficient)) if component_grid.coefficient <= 0 else "+" + str(int(component_grid.coefficient))
                        grid.text(0.55, 0.55, coefficient,
                              fontsize=fontsize * 2, ha='center', color="blue")
                # for axis in ['top', 'bottom', 'left', 'right']:
                #    grid.spines[axis].set_visible(False)
                if operation is not None:
                    operation.plot_component_grid(self, component_grid, grid)
            if True:
                self.plot_outer_axis(fig, linewidth)

        if filename is not None:
            plt.savefig(filename)
        fig.set_tight_layout(False)
        plt.show()
        # reset fontsize to default so it does not affect other figures
        #plt.rcParams.update({'font.size': plt.rcParamsDefault.get('font.size')})
        plt.rcdefaults()
        return fig

    def plot_outer_axis(self, fig, linewidth):
        # fig, overax = plt.subplots()
        # overax = SubplotZero(fig, 111)
        # fig.add_subplot(overax)
        overax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        overax.patch.set_alpha(0)
        # overax.axis('off')
        # overax.set_xticks(np.linspace(0.5/(ncols+1),1 - 0.5/(ncols+1), ncols), range(self.lmin[0], self.lmax[0]+1))
        # overax.set_yticks(np.linspace(0.5/(nrows+1),1 - 0.5/(nrows+1), nrows), range(self.lmin[1], self.lmax[1]+1))
        overax.set_xticks(ticks=[])
        overax.set_yticks(ticks=[])
        overax.set_xlabel("$l_1$")
        overax.set_ylabel("$l_2$")
        overax.set_xlim([0, 1])
        overax.set_ylim([0, 1])
        # plt.rcParams['axes.linewidth'] = 1
        for direction in ["left", "bottom"]:
            # adds arrows at the ends of each axis
            # overax.spines[direction].set_axisline_style("-|>")

            # adds X and Y-axis from the origin
            overax.spines[direction].set_visible(True)
        for direction in ["right", "top"]:
            # hides borders
            overax.spines[direction].set_visible(False)
        overax.arrow(0, 0, 0., 1, fc='k', ec='k', lw=linewidth, head_width=linewidth / 100, head_length=linewidth / 100,
                     overhang=0.3,
                     length_includes_head=True, clip_on=False)
        overax.arrow(0, 0, 1, 0.0, fc='k', ec='k', lw=linewidth, head_width=linewidth / 100,
                     head_length=linewidth / 100, overhang=0.3,
                     length_includes_head=True, clip_on=False)

    def plot_points_component_grid(self, component_grid, consider_not_null, fill_boundary_points, grid, linewidth,
                                   markersize, operation, show_border, ticks):
        points = self.get_points_component_grid(component_grid.levelvector)
        points_not_null = self.get_points_component_grid_not_null(component_grid.levelvector)
        x_array = [p[0] for p in points]
        y_array = [p[1] for p in points]
        x_array_not_null = [[p[0] for p in points_not_null]]
        y_array_not_null = [[p[1] for p in points_not_null]]
        grid.axis('on')
        for axdir in ("x", "y"):
            grid.tick_params(axis=axdir, labelcolor='#345040')
        grid.xaxis.set_ticks_position('none')
        grid.yaxis.set_ticks_position('none')
        if any([math.isinf(x) for x in np.concatenate([self.a, self.b])]):
            offsetx = 0.04 * (max(x_array) - min(x_array))
            offsety = 0.04 * (max(y_array) - min(y_array))
            startx = min(x_array) - offsetx
            starty = min(y_array) - offsety
            endx = max(x_array) + offsetx
            endy = max(y_array) + offsety
            grid.set_xlim([startx, endx])
            grid.set_ylim([starty, endy])
        else:
            startx = self.a[0]
            starty = self.a[1]
            endx = self.b[0]
            endy = self.b[1]
            offsetx = 0.04 * (self.b[0] - self.a[0])
            offsety = 0.04 * (self.b[1] - self.a[1])
            grid.set_xlim([self.a[0] - offsetx, self.b[0] + offsetx])
            grid.set_ylim([self.a[1] - offsety, self.b[1] + offsety])
        if consider_not_null:
            self.plot_points(points=points, grid=grid, markersize=markersize, color="red",
                             fill_boundary=fill_boundary_points)
            self.plot_points(points=points_not_null, grid=grid, markersize=markersize, color="black",
                             fill_boundary=fill_boundary_points)
        else:
            self.plot_points(points=points, grid=grid, markersize=markersize, color="black",
                             fill_boundary=fill_boundary_points)
        grid.spines['top'].set_visible(False)
        grid.spines['right'].set_visible(False)
        grid.spines['bottom'].set_visible(False)
        grid.spines['left'].set_visible(False)
        if show_border and operation is None:
            if component_grid.coefficient != 0:
                facecolor = 'limegreen' if component_grid.coefficient == 1 else 'orange'
            else:
                facecolor = None
            grid.add_patch(
                patches.Rectangle(
                    (startx, starty),
                    endx - startx,
                    endy - starty,
                    fill=facecolor is not None,  # remove background,
                    # alpha=0.5,
                    linewidth=linewidth, visible=True, facecolor=facecolor, edgecolor='black'
                )
            )
        if not ticks:
            grid.axis('off')

    def print_subspaces(self, filename: str=None, add_refinement: bool=True, ticks: bool=True, markersize: int=20, show_border=True, linewidth: float=2.0, show_levelvec: bool=True, fontsize: int=40, figsize: float=10, sparse_grid_spaces: bool=True, fade_full_grid: bool=True, fill_boundary_points=False, consider_not_null: bool=False):
        """This method plots the the subspaces of the generated sparse grid. It might not plot them exactly for adaptive sparse grids.

        :param filename: If set the plot will be set to the specified filename.
        :param add_refinement: If set the refinement structure of the refinement strategy will be plotted.
        :param ticks: If set the ticks in the plots will be set.
        :param markersize: Specifies the marker size in the plot.
        :param show_border: If set the borders of the indivdual plots will be shown for each component grid.
        :param linewidth: Specifies linewidth.
        :param show_levelvec: If set level vectors will be printed above component grids.
        :param fontsize: fontsize that is used for plotting
        :param figsize: unit size of the figure that is generated. This size will be scaled according to the subspaces.
        :param sparse_grid_spaces: if this is set only the subspaces of the sparse grid are shown.
        :param fade_full_grid: if this parameter is set the subspaces that are not in the sparse grid will be colored
        lightgrey.
        Otherwise all full grid spaces are printed.
        :return: Matplotlib Figure.
        """
        plt.rcParams.update({'font.size': fontsize})
        scheme = self.scheme
        lmin = self.lmin
        lmax = self.lmax #[self.combischeme.lmax_adaptive] * self.dim if hasattr(self.combischeme, 'lmax_adaptive') else self.lmax
        dim = self.dim
        if dim != 2:
            self.log_util.log_warning("Cannot print combischeme of dimension > 2")
            return None
        fig, ax = plt.subplots(ncols=self.lmax[0] - self.lmin[0] + 1, nrows=self.lmax[1] - self.lmin[1] + 1, figsize=(figsize*self.lmax[0], figsize*self.lmax[1]))
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
                offsetx = 0.04 * (max(x_array) - min(x_array))
                offsety = 0.04 * (max(y_array) - min(y_array))
                ax.set_xlim([min(x_array) - offsetx, max(x_array) + offsetx])
                ax.set_ylim([min(y_array) - offsety, max(y_array) + offsety])
            else:
                offsetx = 0.04 * (self.b[0] - self.a[0])
                offsety = 0.04 * (self.b[1] - self.a[1])
                ax.set_xlim([self.a[0] - offsetx, self.b[0] + offsetx])
                ax.set_ylim([self.a[1] - offsety, self.b[1] + offsety])
            self.plot_points(points=points, grid=ax, markersize=markersize, color="black",
                             fill_boundary=fill_boundary_points)
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
            if sparse_grid_spaces:
                if self.combischeme.initialized_adaptive:
                    combischeme = self.combischeme
                else:
                    combischeme = CombiScheme(self.dim)
                    combischeme.init_adaptive_combi_scheme(self.lmax[0], self.lmin[0])
            else:
                combischeme = CombiScheme(self.dim)
                combischeme.init_full_grid(self.lmax[0], self.lmin[0])
            levelvectors = combischeme.get_index_set()
            for levelvector in levelvectors:
                points = self.get_points_component_grid(levelvector)
                points_not_null = self.get_points_component_grid_not_null(levelvector)
                x_array = [p[0] for p in points]
                y_array = [p[1] for p in points]
                x_array_not_null = [[p[0] for p in points_not_null]]
                y_array_not_null = [[p[1] for p in points_not_null]]
                grid = ax[lmax[1] - lmin[1] - (levelvector[1] - lmin[1]), (levelvector[0] - lmin[0])]
                grid.axis('on')
                for axdir in ("x", "y"):
                    grid.tick_params(axis=axdir, labelcolor='#345040')
                grid.xaxis.set_ticks_position('none')
                grid.yaxis.set_ticks_position('none')
                if any([math.isinf(x) for x in np.concatenate([self.a, self.b])]):
                    offsetx = 0.04 * (max(x_array) - min(x_array))
                    offsety = 0.04 * (max(y_array) - min(y_array))
                    startx = min(x_array) - offsetx
                    starty = min(y_array) - offsety
                    endx = max(x_array) + offsetx
                    endy =  max(y_array) + offsety
                    grid.set_xlim([startx, endx])
                    grid.set_ylim([starty,endy])
                else:
                    startx = self.a[0]
                    starty = self.a[1]
                    endx = self.b[0]
                    endy =  self.b[1]
                    offsetx = 0.04 * (self.b[0] - self.a[0])
                    offsety = 0.04 * (self.b[1] - self.a[1])
                    grid.set_xlim([self.a[0] - offsetx, self.b[0] + offsetx])
                    grid.set_ylim([self.a[1] - offsety, self.b[1] + offsety])
                # to plot subspaces we need to filter points from lower subspaces
                # filter points from grid to the left (x-1)
                levelvector_x_1 = list(levelvector)
                if levelvector_x_1[0] > self.lmin[0]:
                    levelvector_x_1[0] -= 1
                    points_x1 = self.get_points_component_grid(levelvector_x_1)
                    points_not_null = set(points_not_null) - set(points_x1)
                    points = set(points) - set(points_x1)
                # filter points from grid downwards (y-1)
                levelvector_y_1 = list(levelvector)
                if levelvector_y_1[1] > self.lmin[1]:
                    levelvector_y_1[1] -= 1
                    points_y1 = self.get_points_component_grid(levelvector_y_1)
                    points_not_null = set(points_not_null) - set(points_y1)
                    points = set(points) - set(points_y1)
                if sum(levelvector) > self.lmax[0] + (self.dim - 1) * self.lmin[0] and fade_full_grid:
                    color = 'lightgrey'
                else:
                    color = 'black'
                if consider_not_null:
                    self.plot_points(points=points, grid=grid, markersize=markersize, color="red", fill_boundary=fill_boundary_points)
                    self.plot_points(points=points_not_null, grid=grid, markersize=markersize, color=color, fill_boundary=fill_boundary_points)
                else:
                    self.plot_points(points=points, grid=grid, markersize=markersize, color=color, fill_boundary=fill_boundary_points)
                grid.spines['top'].set_visible(False)
                grid.spines['right'].set_visible(False)
                grid.spines['bottom'].set_visible(False)
                grid.spines['left'].set_visible(False)
                if show_levelvec:
                    grid.set_title(str(tuple(levelvector)))
                if show_border:
                    grid.add_patch(
                        patches.Rectangle(
                            (startx, starty),
                            endx - startx,
                            endy - starty,
                            fill=False,  # remove background,
                            #alpha=0.5,
                            linewidth=linewidth, visible=True, edgecolor=color
                        )
                    )
                if not ticks:
                    grid.axis('off')
                if add_refinement:
                    self.add_refinment_to_figure_axe(grid, linewidth=linewidth)

                if True:
                    #fig, overax = plt.subplots()
                    #overax = SubplotZero(fig, 111)
                    #fig.add_subplot(overax)

                    overax = fig.add_axes([0.1,0.1,0.8,0.8])
                    overax.patch.set_alpha(0)
                    #overax.axis('off')
                    #overax.set_xticks(np.linspace(0.5/(ncols+1),1 - 0.5/(ncols+1), ncols), range(self.lmin[0], self.lmax[0]+1))
                    #overax.set_yticks(np.linspace(0.5/(nrows+1),1 - 0.5/(nrows+1), nrows), range(self.lmin[1], self.lmax[1]+1))
                    overax.set_xticks([],[])
                    overax.set_yticks([],[])
                    overax.set_xlabel("$l_1$")
                    overax.set_ylabel("$l_2$")
                    #plt.rcParams['axes.linewidth'] = 1
                    for direction in ["left", "bottom"]:
                        # adds arrows at the ends of each axis
                        #overax.spines[direction].set_axisline_style("-|>")

                        # adds X and Y-axis from the origin
                        overax.spines[direction].set_visible(True)
                    for direction in ["right", "top"]:
                        # hides borders
                        overax.spines[direction].set_visible(False)
                    overax.arrow(0, 0, 0., 1, fc='k', ec='k', lw = linewidth, head_width=linewidth/100, head_length=linewidth/100, overhang = 0.3,
                    length_includes_head= True, clip_on = False)
                    overax.arrow(0, 0, 1, 0.0, fc='k', ec='k', lw = linewidth, head_width=linewidth/100, head_length=linewidth/100, overhang = 0.3,
                    length_includes_head= True, clip_on = False)


                # for axis in ['top', 'bottom', 'left', 'right']:
                #    grid.spines[axis].set_visible(False)
        # ax1 = fig.add_subplot(111, alpha=0)
        # ax1.set_ylim([self.lmin[1] - 0.5, self.lmax[1] + 0.5])
        # ax1.set_xlim([self.lmin[0] - 0.5, self.lmax[0] + 0.5])
        #plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
        plt.show()
        plt.rcdefaults()
        return fig

    def plot_points(self, points, grid, markersize, color="black", fill_boundary="False"):
        if not fill_boundary:
            points_interior = [p for p in points if not self.grid.point_on_boundary(p)]
            points_boundary = [p for p in points if self.grid.point_on_boundary(p)]
            x_array_interior = [p[0] for p in points_interior]
            y_array_interior = [p[1] for p in points_interior]
            x_array_boundary = [p[0] for p in points_boundary]
            y_array_boundary = [p[1] for p in points_boundary]
            grid.plot(x_array_interior, y_array_interior, 'o', markersize=markersize, color=color)
            grid.plot(x_array_boundary, y_array_boundary, 'o', markersize=markersize, color=color, fillstyle='none')
        else:
            x_array = [p[0] for p in points]
            y_array = [p[1] for p in points]
            grid.plot(x_array, y_array, 'o', markersize=markersize, color=color)

    def print_resulting_sparsegrid(self, filename: str=None, show_fig: bool=True, add_refinement: bool=True, markersize: int=30,
                                   linewidth: float=2.5, ticks: bool=True, color: str="black", show_border: bool=False, figsize: float=20, fill_boundary_points: bool=False, additional_points=[], fontsize: int=60):
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
        plt.rcParams.update({'font.size': fontsize})
        scheme = self.scheme
        dim = self.dim
        if dim != 2 and dim != 3:
            self.log_util.log_warning("Cannot print sparse grid of dimension > 3")
            return None
        if dim == 2:
            fig, ax = plt.subplots(figsize=(figsize, figsize))
            xArray = [p[0] for p in additional_points]
            yArray = [p[1] for p in additional_points]
            plt.plot(xArray, yArray, 'o', markersize=markersize, color="blue")
        if dim == 3:
            fig = plt.figure(figsize=(20, 20))
            ax = fig.add_subplot(111, projection='3d')
            xArray = [p[0] for p in additional_points]
            yArray = [p[1] for p in additional_points]
            zArray = [p[2] for p in additional_points]
            plt.plot(xArray, yArray, zArray, 'o', markersize=markersize, color=color)

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
            offsetx = 0.04 * (end[0] - start[0])
            offsety = 0.04 * (end[1] - start[1])

            ax.set_xlim([start[0] - offsetx, end[0] + offsetx])
            ax.set_ylim([start[1] - offsety, end[1] + offsety])
            if dim == 3:
                ax.set_zlim([start[2] - 0.05, end[2] + 0.05])
        else:
            offsetx = 0.04 * (self.b[0] - self.a[0])
            offsety = 0.04 * (self.b[1] - self.a[1])
            ax.set_xlim([self.a[0] - offsetx, self.b[0] + offsetx])
            ax.set_ylim([self.a[1] - offsety, self.b[1] + offsety])
            if dim == 3:
                ax.set_zlim([self.a[2] - 0.05, self.b[2] + 0.05])
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        if dim == 3:
            ax.zaxis.set_ticks_position('none')
            markersize /= 2

        # get points of each component grid and plot them in one plot
        points = set()
        for component_grid in scheme:
            points = set(self.get_points_component_grid(component_grid.levelvector)) | points

        if dim == 2:
            self.plot_points(points, grid=plt, markersize=markersize, color=color, fill_boundary=fill_boundary_points)
        if dim == 3:
            xArray = [p[0] for p in points]
            yArray = [p[1] for p in points]
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
        # reset fontsize to default so it does not affect other figures
        plt.rcdefaults()
        #plt.rcParams.update({'font.size': plt.rcParamsDefault.get('font.size')})
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
                self.log_util.log_error("{0} Failed for: {1} with value: {2}".format(dictionary, key, value))
                for area in self.refinement.get_objects():
                    self.log_util.log_error("area dict {0}".format(area.levelvec_dict))
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
            self.log_util.log_warning("Grid does not support surplusses")
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
            logUtil.log_error("Dill library not found! Please install dill using pip3 install dill.")

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
            logUtil.log_error("Dill library not found! Please install dill using pip3 install dill.")
