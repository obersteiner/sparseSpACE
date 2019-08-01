import matplotlib.pyplot as plt
import abc
from Function import FunctionShift
import logging
from combiScheme import *
from GridOperation import *
import importlib

# T his class implements the standard combination technique
class StandardCombi(object):
    # initialization
    # a = lower bound of integral; b = upper bound of integral
    # grid = specified grid (e.g. Trapezoidal);
    def __init__(self, a, b, grid=None):
        self.log = logging.getLogger(__name__)
        self.dim = len(a)
        self.a = a
        self.b = b
        self.grid = grid
        self.combischeme = CombiScheme(self.dim)
        assert (len(a) == len(b))

    def __call__(self, interpolation_points):
        interpolation = np.zeros((len(interpolation_points), self.f.output_length()))
        for component_grid in self.scheme:
            interpolation += self.interpolate_points(interpolation_points, component_grid) * component_grid.coefficient
        return interpolation

    def interpolate_points(self, interpolation_points, component_grid: ComponentGridInfo):
        self.grid.setCurrentArea(start=self.a, end=self.b, levelvec=component_grid.levelvector)
        return Interpolation.interpolate_points(f=self.f, dim=self.dim, grid=self.grid, mesh_points_grid=self.grid.coordinate_array, evaluation_points=interpolation_points)

    def interpolate_grid(self, grid):
        num_points = np.prod([len(grid_d) for grid_d in grid])
        interpolation = np.zeros((num_points, self.f.output_length()))
        for component_grid in self.scheme:
            interpolation += self.interpolate_grid_component(grid, component_grid) * component_grid.coefficient
        return interpolation

    def interpolate_grid_component(self, grid, component_grid: ComponentGridInfo):
        grid_points = list(get_cross_product(grid))
        return interpn(grid_points, component_grid)

    def plot(self):
        if self.dim != 2:
            print("Can only plot 2D results")
            return
        xArray = np.linspace(self.a[0], self.b[0], 10 ** 2)
        yArray = np.linspace(self.a[1], self.b[1], 10 ** 2)
        X = [x for x in xArray]
        Y = [y for y in yArray]
        points = list(get_cross_product([X, Y]))
        # print(points)
        #f_values = np.asarray(self.interpolate_grid([X,Y]))

        X, Y = np.meshgrid(X, Y, indexing="ij")
        Z = np.zeros(np.shape(X))
        f_values = np.asarray((self(points)))
        # print(f_values)
        for i in range(len(X)):
            for j in range(len(X[i])):
                # print(X[i,j],Y[i,j],self.eval((X[i,j],Y[i,j])))
                Z[i, j] = f_values[j + i * len(X)]
        # Z=self.eval((X,Y))
        # print Z
        fig = plt.figure(figsize=(14, 6))

        # `ax` is a 3D-aware axis instance, because of the projection='3d' keyword argument to add_subplot
        ax = fig.add_subplot(1, 2, 1, projection='3d')

        # p = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        p = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False)
        # plt.show()
        plt.show()

    def set_combi_parameters(self, minv, maxv, f):
        # compute minimum and target level vector
        self.lmin = [minv for i in range(self.dim)]
        self.lmax = [maxv for i in range(self.dim)]
        # get combi scheme
        self.scheme = self.combischeme.getCombiScheme(minv, maxv)
        self.f = f

    # standard combination scheme for quadrature
    # lmin = minimum level; lmax = target level
    # f = function to integrate; dim=dimension of problem
    def perform_combi(self, minv, maxv, f, reference_solution=None):
        start = self.a
        end = self.b
        self.set_combi_parameters(minv, maxv, f)
        self.f.reset_dictionary()
        combiintegral = 0
        for component_grid in self.scheme:
            integral = self.grid.integrate(self.f, component_grid.levelvector, start, end) * component_grid.coefficient
            combiintegral += integral
        real_integral = reference_solution
        print("CombiSolution", combiintegral)
        if reference_solution is not None:
            print("Analytic Solution", real_integral)
            print("Difference", abs(combiintegral - real_integral))
            return self.scheme, max(abs(combiintegral - real_integral)), combiintegral
        else:
            return self.scheme, None, combiintegral

    def get_num_points_component_grid(self, levelvector, doNaive, num_sub_diagonal):
        return np.prod(self.grid.levelToNumPoints(levelvector))

    # calculate the total number of points used in the complete combination scheme
    def get_total_num_points(self, doNaive=False,
                             distinct_function_evals=True):  # we assume here that all lmax entries are equal
        if distinct_function_evals:
            return self.f.get_f_dict_size()
        numpoints = 0
        for component_grid in self.scheme:
            num_sub_diagonal = (self.lmax[0] + self.dim - 1) - np.sum(component_grid.levelvector)
            pointsgrid = self.get_num_points_component_grid(component_grid.levelvector, doNaive, num_sub_diagonal)
            if distinct_function_evals and self.grid.isNested():
                numpoints += pointsgrid * int(component_grid.coefficient)
            else:
                numpoints += pointsgrid
        # print(numpoints)
        return numpoints

    # prints every single component grid of the combination and orders them according to levels
    def print_resulting_combi_scheme(self, filename=None, add_refinement=True, ticks=True, markersize=20):
        fontsize = 60
        plt.rcParams.update({'font.size': fontsize})
        scheme = self.scheme
        lmin = self.lmin
        lmax = [self.combischeme.lmax_adaptive] * self.dim if hasattr(self.combischeme, 'lmax_adaptive') else self.lmax
        dim = self.dim
        if dim != 2:
            print("Cannot print combischeme of dimension > 2")
            return None
        fig, ax = plt.subplots(ncols=lmax[0] - lmin[0] + 1, nrows=lmax[1] - lmin[1] + 1, figsize=(20, 20))
        # for axis in ax:
        #    spine = axis.spines.values()
        #    spine.set_visible(False)
        # get points of each component grid and plot them individually
        if lmax == lmin:
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.set_xlim([self.a[0] - 0.05, self.b[0] + 0.05])
            ax.set_ylim([self.a[1] - 0.05, self.b[1] + 0.05])
            num_sub_diagonal = (self.lmax[0] + dim - 1) - np.sum(lmax)
            points = self.get_points_component_grid(lmax, num_sub_diagonal)
            x_array = [p[0] for p in points]
            y_array = [p[1] for p in points]
            ax.plot(x_array, y_array, 'o', markersize=markersize, color="black")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            if not ticks:
                ax.axis('off')
            if add_refinement:
                self.add_refinment_to_figure_axe(ax, linewidth=2.0)
        else:

            for i in range(lmax[0] - lmin[0] + 1):
                for j in range(lmax[1] - lmin[1] + 1):
                    ax[i, j].axis('off')

            for component_grid in scheme:
                num_sub_diagonal = (self.lmax[0] + dim - 1) - np.sum(component_grid.levelvector)
                points = self.get_points_component_grid(component_grid.levelvector, num_sub_diagonal)
                points_not_null = self.get_points_component_grid_not_null(component_grid.levelvector, num_sub_diagonal)
                x_array = [p[0] for p in points]
                y_array = [p[1] for p in points]
                x_array_not_null = [[p[0] for p in points_not_null]]
                y_array_not_null = [[p[1] for p in points_not_null]]
                grid = ax[lmax[1] - lmin[1] - (component_grid.levelvector[1] - lmin[1]), (component_grid.levelvector[0] - lmin[0])]
                grid.axis('on')
                grid.xaxis.set_ticks_position('none')
                grid.yaxis.set_ticks_position('none')
                grid.set_xlim([self.a[0] - 0.05, self.b[0] + 0.05])
                grid.set_ylim([self.a[1] - 0.05, self.b[1] + 0.05])
                grid.plot(x_array, y_array, 'o', markersize=markersize, color="red")
                grid.plot(x_array_not_null, y_array_not_null, 'o', markersize=markersize, color="black")
                grid.spines['top'].set_visible(False)
                grid.spines['right'].set_visible(False)
                grid.spines['bottom'].set_visible(False)
                grid.spines['left'].set_visible(False)
                if not ticks:
                    grid.axis('off')
                if add_refinement:
                    self.add_refinment_to_figure_axe(grid, linewidth=2.0)

                coefficient = str(int(component_grid.coefficient)) if component_grid.coefficient <= 0 else "+" + str(int(component_grid.coefficient))
                grid.text(0.55, 0.55, coefficient,
                          fontsize=fontsize * 2, ha='center', color="blue")
                # for axis in ['top', 'bottom', 'left', 'right']:
                #    grid.spines[axis].set_visible(False)
        # ax1 = fig.add_subplot(111, alpha=0)
        # ax1.set_ylim([self.lmin[1] - 0.5, self.lmax[1] + 0.5])
        # ax1.set_xlim([self.lmin[0] - 0.5, self.lmax[0] + 0.5])

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
        plt.show()
        return fig

    # prints the sparse grid which results from the combination
    def print_resulting_sparsegrid(self, filename=None, show_fig=True, add_refinement=True, markersize=30,
                                   linewidth=2.5, ticks=True, color="black"):
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

        ax.set_xlim([self.a[0] - 0.05, self.b[0] + 0.05])
        ax.set_ylim([self.a[1] - 0.05, self.b[1] + 0.05])
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        if dim == 3:
            ax.set_zlim([self.a[2] - 0.05, self.b[2] + 0.05])
            ax.zaxis.set_ticks_position('none')
            markersize /= 2

        # get points of each component grid and plot them in one plot
        for component_grid in scheme:
            numSubDiagonal = (self.lmax[0] + dim - 1) - np.sum(component_grid.levelvector)
            points = self.get_points_component_grid(component_grid.levelvector, numSubDiagonal)
            xArray = [p[0] for p in points]
            yArray = [p[1] for p in points]
            if dim == 2:
                plt.plot(xArray, yArray, 'o', markersize=markersize, color=color)
            if dim == 3:
                zArray = [p[2] for p in points]
                plt.plot(xArray, yArray, zArray, 'o', markersize=markersize, color=color)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
        if not ticks:
            ax.axis('off')
        if add_refinement and dim == 2:
            self.add_refinment_to_figure_axe(ax, linewidth=linewidth)
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
        if show_fig:
            plt.show()
        return fig

    # check if combischeme is right
    def check_combi_scheme(self):
        if not self.grid.isNested():
            return
        dim = self.dim
        dictionary = {}
        for component_grid in self.scheme:
            num_sub_diagonal = (self.lmax[0] + dim - 1) - np.sum(component_grid.levelvector)
            # print num_sub_diagonal , ii ,component_grid
            points = self.get_points_component_grid_not_null(component_grid.levelvector, num_sub_diagonal)
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
                '''
                for area in self.refinement.getObjects():
                    print("new area:",area)
                    for component_grid in self.scheme:
                        num_sub_diagonal = (self.lmax[0] + dim - 1) - np.sum(component_grid[0])
                        self.coarsenGrid(component_grid[0],area, num_sub_diagonal,key)
                #print(self.refinement)
                #print(dictionary.items())
                '''
            assert (value == 1)

    def get_points_component_grid_not_null(self, levelvec, numSubDiagonal) -> Sequence[Tuple[float, ...]]:
        return self.get_points_component_grid(levelvec, numSubDiagonal)

    def get_points_component_grid(self, levelvec, numSubDiagonal) -> Sequence[Tuple[float, ...]]:
        self.grid.setCurrentArea(self.a, self.b, levelvec)
        points = self.grid.getPoints()
        return points

    def get_points_and_weights_component_grid(self, levelvec, numSubDiagonal) -> Tuple[Sequence[Tuple[float, ...]], Sequence[float]]:
        self.grid.setCurrentArea(self.a, self.b, levelvec)
        return self.grid.get_points_and_weights()

    def get_points_and_weights(self) -> Tuple[Sequence[Tuple[float, ...]], Sequence[float]]:
        total_points = []
        total_weights = []
        for component_grid in self.scheme:
            num_sub_diagonal = (self.lmax[0] + self.dim - 1) - np.sum(component_grid.levelvector)
            points, weights = self.get_points_and_weights_component_grid(component_grid.levelvector, num_sub_diagonal)
            total_points.extend(points)
            # adjust weights for combination -> multiply with combi coefficient
            weights = [w * component_grid.coefficient for w in weights]
            total_weights.extend(weights)
        return np.asarray(total_points), np.asarray(total_weights)

    def get_surplusses(self) -> Sequence[Sequence[float]]:
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

    def add_refinment_to_figure_axe(self, ax, linewidth=1):
        pass

    @staticmethod
    def restore_from_file(fileName):
        spam_spec = importlib.util.find_spec("dill")
        found = spam_spec is not None
        if found:
            import dill
            with open(fileName, 'rb') as f:
                return dill.load(f)
        else:
            print("Dill library not found! Please install dill using pip3 install dill.")

    def save_to_file(self, fileName):
        spam_spec = importlib.util.find_spec("dill")
        found = spam_spec is not None
        if found:
            import dill
            with open(fileName, 'wb') as f:
                dill.dump(self, f)
        else:
            print("Dill library not found! Please install dill using pip3 install dill.")