import scipy.integrate
import numpy as np
import abc,logging


#This class is the general interface of an error estimator currently used by the algorithm
class ErrorCalculator(object):
    #initialization
    def __init__(self):
        self.log=logging.getLogger(__name__)
    #calculates error for the function f and the integral information that was computed by the algorithm
    #this information contians the area specification and the approximated integral
    #current form is (approxIntegral,start,end)
    @abc.abstractmethod
    def calcError(self,f,refineObject):
        return
#This error estimator doea a comparison to analytic solution. It outputs the absolut error.
class ErrorCalculatorAnalytic(ErrorCalculator):
    def calcError(self,f, refineObject):
        lowerBounds = refineObject.start
        upperBounds = refineObject.end
        dim = len(lowerBounds)
        realIntegralValue = f.getAnalyticSolutionIntegral(lowerBounds,upperBounds)
        return abs(refineObject.integral - realIntegralValue)

    def calcErrorArrayToRefinement(self,f,refineObject):
        lowerBounds = refineObject.start
        upperBounds = AreaInfo.end
        dim = len(lowerBounds)
        realIntegralValue = f.getAnalyticSolutionIntegral(lowerBounds,upperBounds)
        return abs(np.sum(integralarrayPartial) - realIntegralValue)

#This error estimator doea a comparison to analytic solution. It outputs the relative error.
class ErrorCalculatorAnalyticRelative(ErrorCalculator):
    def __init__(self,a,b):
        self.a = a
        self.b = b
    def calcError(self,f, refineObject):
        lowerBounds = refineObject.start
        upperBounds = refineObject.end
        dim = len(lowerBounds)
        realIntegralValue = f.getAnalyticSolutionIntegral(lowerBounds,upperBounds)
        realIntegralComplete = f.getAnalyticSolutionIntegral(a,b)
        return abs((refineObject.integral - realIntegralValue)/realIntegralComplete)