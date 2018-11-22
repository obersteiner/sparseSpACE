from spatiallyAdaptiveBase import *


class SpatiallyAdaptiveFixedScheme(SpatiallyAdaptivBase):

    # returns the points of a single component grid with refinement
    def get_points_component_grid(self, levelvec, numSubDiagonal):
        dim = len(levelvec)
        array2 = []
        for area in self.refinement.get_objects():
            start = area.start
            end = area.end
            levelInterval = np.zeros(dim, dtype=int)
            for d in range(dim):
                levelInterval[d] = int(levelvec[d] - self.lmin[d])
            self.grid.setCurrentArea(start, end, levelInterval)
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
        fig = plt.figure(figsize=(20, 20))
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
                    endx - startx,
                    endy - starty,
                    fill=False  # remove background
                )
            )
            # content = str(i[5])
            xcontent = startx + (endx - startx) / 2.0
            ycontent = starty + (endy - starty) / 2.0
            # print content, xcontent, ycontent
            # ax.annotate(content, (xcontent,ycontent))
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
        plt.show()
        return fig

    def initialize_refinement(self):
        new_refinement_objects = RefinementObjectExtendSplit(np.array(self.a), np.array(self.b), 1000, 0,
                                                             0).split_area_arbitrary_dim()
        self.refinement = RefinementContainer(new_refinement_objects, self.dim, self.errorEstimator)

    def evaluate_area(self, f, area, levelvec):
        levelEval = np.zeros(self.dim, dtype=int)
        for d in range(self.dim):
            levelEval[d] = int(levelvec[d] - self.lmin[d])
        return self.grid.integrate(f, levelEval, area.start, area.end), None, np.prod(
            self.grid.levelToNumPoints(levelEval))

    def do_refinement(self, area, position):
        self.refinement.refine(position)
        return False
