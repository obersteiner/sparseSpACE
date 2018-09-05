# from RefinementContainer import *
from spatiallyAdaptiveBase import *


class SpatiallyAdaptivCellScheme(SpatiallyAdaptivBase):
    def __init__(self, a, b, grid=TrapezoidalGrid):
        SpatiallyAdaptivBase.__init__(self, a, b, grid)
        # dummy container
        self.refinement = RefinementContainer([], self.dim, None)

    # returns the points of a single component grid with refinement
    def get_points_arbitrary_dim(self, levelvec, numSubDiagonal):
        dim = len(levelvec)
        array2 = []
        for area in self.refinement.get_objects():
            start = area.start
            end = area.end
            self.grid.setCurrentArea(start, end, np.zeros(dim))
            points = self.grid.getPoints()
            array2.extend(points)
        # print array2
        return array2

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
        initial_object = RefinementObjectCell(np.array(self.a), np.array(self.b), -(self.dim-1))
        new_refinement_objects = initial_object.split_cell_arbitrary_dim().append(initial_object)
        self.refinement = RefinementContainer(new_refinement_objects, self.dim, self.errorEstimator)

    def evaluate_area(self, f, area, levelvec):
        level_eval = np.zeros(self.dim, dtype=int)
        return self.grid.integrate(f, level_eval, area.start, area.end), None, np.prod(self.grid.levelToNumPoints(level_eval))

    def do_refinement(self, area, position):
        self.refinement.refine(position)
        return False
