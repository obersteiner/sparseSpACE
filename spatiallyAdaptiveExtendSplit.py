
from spatiallyAdaptiveBase import *

class SpatiallyAdaptivExtendScheme(SpatiallyAdaptivBase):
    def __init__(self,a,b,numberOfRefinementsBeforeExtend,grid=TrapezoidalGrid(),noInitialSplitting=False, topDiagIncreaseCoarsening=False):
        SpatiallyAdaptivBase.__init__(self,a,b,grid)
        self.noInitialSplitting = noInitialSplitting
        self.numberOfRefinementsBeforeExtend = numberOfRefinementsBeforeExtend
        self.topDiagIncreaseCoarsening = topDiagIncreaseCoarsening

    #draw a visual representation of refinement tree
    def drawRefinement(self,filename = None):
        plt.rcParams.update({'font.size': 32})
        dim = self.dim
        if(dim>2):
            print("Refinement can only be printed in 2D")
            return
        fig = plt.figure(figsize=(20,20))
        ax2 = fig.add_subplot(111, aspect='equal')
        #print refinement
        for i in self.refinement.getObjects():
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
            #content = str(i[5])
            xcontent = startx+(endx-startx)/2.0
            ycontent = starty+(endy-starty)/2.0
            #print content, xcontent, ycontent
            #ax.annotate(content, (xcontent,ycontent))
        if(filename != None):
            plt.savefig(filename, bbox_inches='tight')
        plt.show()
        return fig

    #returns the points of a single component grid with refinement
    def getPointsArbitraryDim(self,levelvec, numSubDiagonal):
        assert(numSubDiagonal < self.dim)
        array2 = []
        for area in self.refinement.getObjects():
            start = area.start
            end = area.end
            levelInterval = self.coarsenGrid(levelvec,area,numSubDiagonal)
            #print(levelInterval,levelvec,start,end)
            self.grid.setCurrentArea(start,end,levelInterval)
            points = self.grid.getPoints()
            array2.extend(points)
        return array2

    #optimized adaptive refinement refine multiple cells in close range around max variance (here set to 10%)
    def coarsenGrid(self,levelvector,area, numSubDiagonal,printPoint=None):
        start = area.start
        end = area.end
        coarsening = area.coarseningValue
        temp = list(levelvector)
        coarseningSave = coarsening
        #print(coarsening)
        while(coarsening > 0):
            maxLevel = max(temp)
            if maxLevel == self.lmin[0]: #we assume here that lmin is equal everywhere
                break
            occurencesOfMax = 0
            for i in temp:
                if i == maxLevel:
                    occurencesOfMax += 1
            isTopDiag = numSubDiagonal == 0
            if(self.topDiagIncreaseCoarsening):
                noForwardProblem = coarseningSave >= self.lmax[0] + self.dim - 1 - maxLevel - (self.dim-2) - maxLevel + 1
                doCoarsen = noForwardProblem and coarsening >= occurencesOfMax - isTopDiag
            else:
                noForwardProblem = coarseningSave >= self.lmax[0] + self.dim - 1 - maxLevel - (self.dim-2) - maxLevel + 2
                doCoarsen = noForwardProblem and coarsening >= occurencesOfMax
            if(doCoarsen):
                for d in range(self.dim):
                    if temp[d] == maxLevel:
                        temp[d] -= 1
                        coarsening -= 1
            else:
                break
        levelCoarse = [ temp[d]-self.lmin[d]+int(self.noInitialSplitting) for d in range(len(temp))]
        if(printPoint != None):
            if(all([start[d] <= printPoint[d] and end[d] >= printPoint[d] for d in range(self.dim)])):
                print("Level: ", levelvector, "Coarsened level:", levelCoarse, coarseningSave, start, end)
        return levelCoarse

    def initializeRefinement(self):
        if(self.noInitialSplitting):
            newRefinementObject = RefinementObjectExtendSplit(np.array(self.a),np.array(self.b), self.numberOfRefinementsBeforeExtend, 0, 0)
            self.refinement = RefinementContainer([newRefinementObject], self.dim, self.errorEstimator)
        else:
            newRefinementObjects = RefinementObjectExtendSplit(np.array(self.a),np.array(self.b), self.numberOfRefinementsBeforeExtend, 0, 0).splitAreaArbitraryDim()
            self.refinement = RefinementContainer(newRefinementObjects,self.dim, self.errorEstimator)

    def evaluateArea(self, f,area,levelvec):
        numSubDiagonal = (self.lmax[0] + self.dim - 1) - np.sum(levelvec)
        levelForEvaluation = self.coarsenGrid(levelvec,area,numSubDiagonal)
        return self.grid.integrate(f,levelForEvaluation,area.start,area.end), None, np.prod(self.grid.levelToNumPoints(levelForEvaluation))


    def doRefinement(self,area,position):
        lmaxChange = self.refinement.refine(position)
        if lmaxChange != None:
            self.lmax = [self.lmax[d] + lmaxChange[d] for d in range(self.dim)]
            print("New scheme")
            self.scheme = getCombiScheme(self.lmin[0],self.lmax[0],self.dim)
            return True
        return False