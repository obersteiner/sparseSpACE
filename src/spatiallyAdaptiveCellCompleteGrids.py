# from RefinementContainer import *
from spatiallyAdaptiveBase import *
import itertools
from scipy.interpolate import griddata
from scipy.interpolate import interpn

class SpatiallyAdaptivCellSchemeGrid(SpatiallyAdaptivBase):
    def __init__(self, a, b, grid=TrapezoidalGrid, punish_depth=False):
        SpatiallyAdaptivBase.__init__(self, a, b, grid)
        RefinementObjectCell.punish_depth = punish_depth
        # dummy container
        self.refinement = RefinementContainer([], self.dim, None)
        self.max_level = np.ones(self.dim)
        self.full_interaction_size = 1
        self.level_to_cell_dict = {}
        for d in range(1, self.dim+1):
            self.full_interaction_size += math.factorial(self.dim)/(math.factorial(d)*math.factorial(self.dim - d)) * 2**d
        #print("full interaction size:", self.full_interaction_size)
    # returns the points of a single component grid with refinement
    def get_points_component_grid(self, levelvec, numSubDiagonal):
        cells = self.level_to_cell_dict[tuple(levelvec)]
        points = set()
        for cell in cells:
            points = points | set(self.get_points_cell(cell))
        return self.grid_points

    # draw a visual representation of refinement tree
    def draw_refinement(self, filename=None):
        plt.rcParams.update({'font.size': 32})
        dim = len(self.refinement[0][0])
        if dim > 2:
            print("Refinement can only be printed in 2D")
            return
        fig = plt.figure(figsize=(20,20))
        ax2 = fig.add_subplot(111, aspect='equal')
        # print refinement
        for i in self.refinement.get_objects():
            startx = i.start[0]
            starty = i.start[1]
            endx = i.end[0]
            endy = i.end[1]
            ax2.add_patch(
                patches.Rectangle(
                    (startx, starty),
                    endx-startx,
                    endy-starty,
                    fill=False      # remove background
                )
            )
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
        plt.show()
        return fig

    def initialize_refinement(self):
        RefinementObjectCell.cell_dict = {}
        CombiScheme.initialized_adaptive(self.dim, self.lmin[0], self.lmax[0])
        CombiScheme.dim = self.dim
        CombiScheme.lmin = self.lmin[0]
        assert self.lmin == self.lmax
        #CombiScheme.init_adaptive_combi_scheme(self.dim, self.lmax, self.lmin)
        self.rootCell = RefinementObjectCell(np.array(self.a), np.array(self.b), np.zeros(self.dim), self.a, self.b, self.lmin)
        initial_objects = [self.rootCell]
        for d in range(self.dim):
            for l in range(self.lmin[d]):
                print("split dimension", d)
                initial_objects = [i.split_cell_arbitrary_dim(d) for i in initial_objects]  # is now a list of lists
                # flatten list again
                initial_objects = list(itertools.chain(*initial_objects))

        self.grid_points = set(zip(*[g.ravel() for g in np.meshgrid(*[np.linspace(self.a[d], self.b[d], 2**self.lmin[d] + 1) for d in range(self.dim)])]))
        self.f_dict = {}
        for p in self.grid_points:
            self.f_dict[p] = self.f.eval(p)
        self.refinement = RefinementContainer(initial_objects, self.dim, self.errorEstimator)
        self.scheme = CombiScheme.getCombiScheme(self.lmin, self.lmax, self.dim)
        self.level_to_cell_dict[tuple(self.scheme[0][0])] = [object.get_key() for object in initial_objects]
        for object in initial_objects:
            object.add_grid(tuple(self.scheme[0][0]))

    def evaluate_area(self, f, area, component_grid):  # area is a cell here
        cells_contained = self.level_to_cell_dict[tuple(component_grid.levelvector)]
        integral = 0
        evaluations = 0
        for cell in cells_contained:
            integral += self.integrate_cell(cell)
            evaluations += 2**self.dim
        relevant_parents_of_cell = [(area.get_key(), area.levelvec, 1)]
        for d in range(self.dim):
            new_parents = []
            for subarea in relevant_parents_of_cell:
                levelvec_parent = list(subarea[1])
                levelvec_parent[d] -= 1
                start_subarea = subarea[0][0]
                end_subarea = subarea[0][1]
                coefficient = subarea[2]
                parent_area = RefinementObjectCell.parent_cell_arbitrary_dim(d, list(subarea[1]), start_subarea, end_subarea, self.a, self.b, self.lmin)
                if parent_area is not None:
                    new_parents.append((parent_area, levelvec_parent, coefficient*-1))
            relevant_parents_of_cell.extend(new_parents)

        #integral = 0
        #evaluations = 0
        #print(len(relevant_parents_of_cell))
        for (parentcell, levelvec, coefficient) in relevant_parents_of_cell:
            #print(area.get_key(), parentcell, coefficient)
            if coefficient != 0:
                sub_integral = self.integrate_subcell_with_interpolation(parentcell, area.get_key())
                RefinementObjectCell.cell_dict[area.get_key()].sub_integrals.append((sub_integral, coefficient))
                #integral += sub_integral * coefficient
                #print(self.integrate_subcell_with_interpolation(area.get_key(), subcell) * coefficient)
                #evaluations += 2**self.dim
        else:
            pass
            #print("Nothing to do in this region")
        #print("integral of cell", area.get_key(), "is:", integral)
        

        return integral, None, evaluations


    def evaluate_area2(self, f, area, levelvec):  # area is a cell here
        subareas_in_cell = [(area.get_key(), area.levelvec, 1)]
        subareas_fused = {}
        subareas_fused[area.get_key()] = 1
        for d in range(self.dim):
            new_subareas = []
            for subarea in subareas_in_cell:
                levelvec_subarea = list(subarea[1])
                levelvec_subarea[d] += 1
                start_subarea = subarea[0][0]
                end_subarea = subarea[0][1]
                coefficient = subarea[2]
                new_areas = RefinementObjectCell.children_cell_arbitrary_dim(d, start_subarea, end_subarea, self.dim)
                new_subareas_refinement = []
                for area_candidate in new_areas:
                    if area_candidate in RefinementObjectCell.cell_dict:
                        new_subareas_refinement.append((area_candidate, levelvec_subarea, (coefficient*-1)))
                new_subareas.extend(new_subareas_refinement)
                #if len(new_subareas_refinement) == len(new_areas):
                #    if subarea[0] in subareas_fused:
                #        subareas_fused[subarea[0]] += coefficient * -1
                #    else:
                #        subareas_fused[subarea[0]] = coefficient * -1
                #else:
                for s in new_subareas_refinement:
                    subareas_fused[s[0]] = s[2]
            subareas_in_cell.extend(new_subareas)

        integral = 0
        evaluations = 0
        if len(subareas_in_cell) != self.full_interaction_size:
            #print(len(subareas_in_cell))
            for subcell, coefficient in subareas_fused.items():
                #print(area.get_key(), subcell, coefficient)
                if coefficient != 0:
                    sub_integral = self.integrate_subcell_with_interpolation(area.get_key(), subcell)
                    RefinementObjectCell.cell_dict[subcell].sub_integrals.append((sub_integral, coefficient))
                    integral += sub_integral * coefficient
                    #print(self.integrate_subcell_with_interpolation(area.get_key(), subcell) * coefficient)
                    evaluations += 2**self.dim
        else:
            pass
            #print("Nothing to do in this region")
        #print("integral of cell", area.get_key(), "is:", integral)
        '''
        for p in self.grid_points:
            print(RefinementObjectCell.cell_dict)
            value = 0
            cells_with_point, levelvec = self.get_cells_to_point(p)
            level_to_cell_dict = {}
            for cell in cells_with_point:
                level_to_cell_dict[cell[1]] = cell[0]
            print(cells_with_point)
            coefficients = CombiScheme.get_coefficients_to_index_set(set([cell[1] for cell in cells_with_point]))
            print(coefficients)
            for i, coefficient in enumerate(coefficients):
                if coefficient[1] != 0:
                value += self.interpolate_point(level_to_cell_dict[coefficient[0]], p) * coefficient[1]
            print("Combined value at position:", p, "is", value, "with levelevector:", levelvec, "function value is", self.f_dict[p])
        '''
        return integral, None, evaluations

    def integrate_subcell_with_interpolation(self, cell, subcell):
        #print("Cell and subcell", cell, subcell)
        start_subcell = subcell[0]
        end_subcell = subcell[1]
        subcell_points = list(zip(*[g.ravel() for g in np.meshgrid(*[[start_subcell[d], end_subcell[d]] for d in range(self.dim)])]))
        interpolated_values = self.interpolate_points(cell, subcell_points)
        width = np.prod(np.array(end_subcell) - np.array(start_subcell))
        factor = 0.5**self.dim * width
        integral = 0.0
        for p in interpolated_values:
            integral += p * factor
        #print("integral of subcell", subcell, "of cell", cell, "is", integral, "interpolated values", interpolated_values, "on points", subcell_points, "factor", factor)
        return integral

    def integrate_cell(self,cell):
        start = cell.start
        end = cell.end
        cell_points = self.get_points_cell(cell)
        cell_values = [self.f_dict[p] for p in cell_points]
        width = np.prod(np.array(end) - np.array(start))
        factor = 0.5**self.dim * width
        integral = 0.0
        for p in cell_values:
            integral += p * factor
        return integral

    def get_points_cell(self, cell):
        return list(zip(*[g.ravel() for g in np.meshgrid(*[[start[d], end_sub[d]] for d in range(self.dim)])]))

    def do_refinement(self, cell, position):
        if cell.active:
            #print("refining object:", area.get_key())
            self.refinement.refine(position)
            CombiScheme.refine_scheme()
            cell.error = 0
        return False

    def refinement_postprocessing(self):
        new_objects = self.refinement.get_new_objects()
        for object in new_objects:
            points_in_object = object.get_points()
            #print(points_in_object)
            for p in points_in_object:
                self.grid_points.add(p)
                self.f_dict[p] = self.f.eval(p)
        #self.refinement.apply_remove()
        self.refinement.refinement_postprocessing()
        #self.refinement.reinit_new_objects()

    def get_areas(self):
        return self.refinement.get_objects()

    def get_new_areas(self):
        #return self.refinement.get_objects()
        return self.refinement.get_new_objects()

    def get_cells_to_point(self, point):
        return self.get_children_with_point(self.rootCell, point)

    def get_children_with_point(self, cell, point):
        #print("cell:", cell.get_key(),"children:", cell.children)
        cell_list = set()
        cell_list.add(tuple((cell.get_key(), tuple(cell.levelvec))))
        if cell.is_corner(point):
            levelvec = cell.levelvec
        else:
            levelvec = None
        for child in cell.children:
            if child.contains(point):
                cell_list_new, levelvecNew = self.get_children_with_point(child, point)
                cell_list = cell_list | cell_list_new
                if levelvec is None or (levelvecNew is not None and levelvecNew <= levelvec):
                    levelvec = levelvecNew
        return cell_list, levelvec
