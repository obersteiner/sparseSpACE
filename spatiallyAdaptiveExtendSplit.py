from spatiallyAdaptiveBase import *


class SpatiallyAdaptivExtendScheme(SpatiallyAdaptivBase):
    def __init__(self, a, b, number_of_refinements_before_extend, grid=TrapezoidalGrid(), no_initial_splitting=False,
                 top_diag_increase_coarsening=False):
        SpatiallyAdaptivBase.__init__(self, a, b, grid)
        self.noInitialSplitting = no_initial_splitting
        self.numberOfRefinementsBeforeExtend = number_of_refinements_before_extend
        self.topDiagIncreaseCoarsening = top_diag_increase_coarsening

    # draw a visual representation of refinement tree
    def draw_refinement(self, filename=None):
        plt.rcParams.update({'font.size': 32})
        dim = self.dim
        if dim > 2:
            print("Refinement can only be printed in 2D")
            return
        fig = plt.figure(figsize=(20, 20))
        ax2 = fig.add_subplot(111, aspect='equal')
        for i in self.refinement.get_objects():
            startx = i.start[0]
            starty = i.start[1]
            endx = i.end[0]
            endy = i.end[1]
            ax2.add_patch(
                patches.Rectangle(
                    (startx, starty),
                    endx - startx,
                    endy - starty,
                    fill=False  # remove background
                )
            )
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
        plt.show()
        return fig

    # returns the points of a single component grid with refinement
    def get_points_arbitrary_dim(self, levelvec, numSubDiagonal):
        assert (numSubDiagonal < self.dim)
        array2 = []
        for area in self.refinement.get_objects():
            start = area.start
            end = area.end
            level_interval = self.coarsen_grid(levelvec, area, numSubDiagonal)
            self.grid.setCurrentArea(start, end, level_interval)
            points = self.grid.getPoints()
            array2.extend(points)
        return array2

    # optimized adaptive refinement refine multiple cells in close range around max variance (here set to 10%)
    def coarsen_grid(self, levelvector, area, num_sub_diagonal, print_point=None):
        start = area.start
        end = area.end
        coarsening = area.coarseningValue
        temp = list(levelvector)
        coarsening_save = coarsening
        while coarsening > 0:
            maxLevel = max(temp)
            if maxLevel == self.lmin[0]:  # we assume here that lmin is equal everywhere
                break
            occurences_of_max = 0
            for i in temp:
                if i == maxLevel:
                    occurences_of_max += 1
            is_top_diag = num_sub_diagonal == 0
            if self.topDiagIncreaseCoarsening:
                no_forward_problem = coarsening_save >= self.lmax[0] + self.dim - 1 - maxLevel - (
                            self.dim - 2) - maxLevel + 1
                do_coarsen = no_forward_problem and coarsening >= occurences_of_max - is_top_diag
            else:
                no_forward_problem = coarsening_save >= self.lmax[0] + self.dim - 1 - maxLevel - (
                            self.dim - 2) - maxLevel + 2
                do_coarsen = no_forward_problem and coarsening >= occurences_of_max
            if do_coarsen:
                for d in range(self.dim):
                    if temp[d] == maxLevel:
                        temp[d] -= 1
                        coarsening -= 1
            else:
                break
        level_coarse = [temp[d] - self.lmin[d] + int(self.noInitialSplitting) for d in range(len(temp))]
        if print_point is not None:
            if all([start[d] <= print_point[d] and end[d] >= print_point[d] for d in range(self.dim)]):
                print("Level: ", levelvector, "Coarsened level:", level_coarse, coarsening_save, start, end)
        return level_coarse

    def initialize_refinement(self):
        if (self.noInitialSplitting):
            new_refinement_object = RefinementObjectExtendSplit(np.array(self.a), np.array(self.b),
                                                              self.numberOfRefinementsBeforeExtend, 0, 0)
            self.refinement = RefinementContainer([new_refinement_object], self.dim, self.errorEstimator)
        else:
            new_refinement_objects = RefinementObjectExtendSplit(np.array(self.a), np.array(self.b),
                                                               self.numberOfRefinementsBeforeExtend, 0,
                                                               0).split_area_arbitrary_dim()
            self.refinement = RefinementContainer(new_refinement_objects, self.dim, self.errorEstimator)

    def evaluate_area(self, f, area, levelvec):
        num_sub_diagonal = (self.lmax[0] + self.dim - 1) - np.sum(levelvec)
        level_for_evaluation = self.coarsen_grid(levelvec, area, num_sub_diagonal)
        return self.grid.integrate(f, level_for_evaluation, area.start, area.end), None, np.prod(
            self.grid.levelToNumPoints(level_for_evaluation))

    def do_refinement(self, area, position):
        lmax_change = self.refinement.refine(position)
        if lmax_change != None:
            self.lmax = [self.lmax[d] + lmax_change[d] for d in range(self.dim)]
            print("New scheme")
            self.scheme = getCombiScheme(self.lmin[0], self.lmax[0], self.dim)
            return True
        return False
