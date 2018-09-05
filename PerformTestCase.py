
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Grid import *
import time
from StandardCombi import *

def performTestStandard(f, a, b, grid, lmin,maxLmax,dim,realIntegral):
    #calculate standard combination scheme results
    errorArrayStandard = []
    pointArray = []
    distinctFEvalArray = []
    standardCombi = StandardCombi(a, b, grid)
    for i in range(lmin+1,maxLmax):
        scheme, error, result = standardCombi.performCombi(lmin,i,f,dim)
        errorArrayStandard.append(error/realIntegral)
        pointArray.append(standardCombi.getTotalNumPoints())
        distinctFEvalArray.append(standardCombi.getTotalNumPoints(True))
    return  pointArray, distinctFEvalArray, errorArrayStandard

def performTestcaseArbitraryDim(f,a,b,adaptiveAlgorithmVector, errorOperator, maxtol, dim,maxLmax,grid = TrapezoidalGrid(),minLmin=1,maxLmin=3):
    #realIntegral = scipy.integrate.dblquad(f, a, b, lambda x:a, lambda x:b, epsabs=1e-15, epsrel=1e-15)[0]
    realIntegral = f.getAnalyticSolutionIntegral(a,b)
    print("Exact integral", realIntegral)
    errorArray = []
    numEvaluationsArray = []
    numNaive = []
    numIdeal = []
    numFEvalIdeal = []
    #calculate refinements for different tolerance values
    for algorithm in adaptiveAlgorithmVector:
        errorArrayAlgorithm = []
        numEvaluationsArrayAlgorithm = []
        numNaiveAlgorithm = []
        numIdealAlgorithm = []
        numFEvalIdealAlgorithm = []
        for i in range(1,maxtol):
            start = time.time()
            if(i == 1):
                coarsening,combischeme,lmax,integral,numberOfEvaluations= algorithm[0].performSpatiallyAdaptiv(algorithm[1],algorithm[2],f,errorOperator,10**-i)
            else:
                if (abs(integral - realIntegral) > 10**-i):
                    coarsening,combischeme,lmax,integral,numberOfEvaluations= algorithm[0].performSpatiallyAdaptiv(algorithm[1],algorithm[2],f,errorOperator,10**-i,coarsening)
            errorArrayAlgorithm.append(abs(integral - realIntegral)/abs(realIntegral))
            numEvaluationsArrayAlgorithm.append(numberOfEvaluations)
            numIdealAlgorithm.append(algorithm[0].getTotalNumPointsArbitraryDim(False))
            numNaiveAlgorithm.append(algorithm[0].getTotalNumPointsArbitraryDim(True))
            numFEvalIdealAlgorithm.append(algorithm[0].getTotalNumPointsArbitraryDim(False,True))
            end = time.time()
            print("time spent in case", i, end - start)
        errorArray.append(errorArrayAlgorithm)
        numEvaluationsArray.append(numEvaluationsArrayAlgorithm)
        numNaive.append(numNaiveAlgorithm)
        numIdeal.append(numIdealAlgorithm)
        numFEvalIdeal.append(numFEvalIdealAlgorithm)

    #calculate different standard combination scheme results
    xArrayStandard = []
    xFEvalArrayStandard = []
    errorArrayStandard  = []
    for i in range(minLmin,maxLmin):
        xArrayStandardTest, xFEvalArrayStandardTest, errorArrayStandardTest = performTestStandard(f, a, b, grid, i, maxLmax-(i-1)*(dim-1),dim,realIntegral)
        xArrayStandard.append(xArrayStandardTest)
        xFEvalArrayStandard.append(xFEvalArrayStandardTest)
        errorArrayStandard.append(errorArrayStandardTest)
    #plot
    for i in range(maxLmin-minLmin):
        print(xArrayStandard[i], errorArrayStandard[i], "Number of Points Standard lmin= " + str(i+minLmin))
        print(xFEvalArrayStandard[i], errorArrayStandard[i], "Distinct f evaks Standard lmin= " + str(i+minLmin))
        plt.loglog(xArrayStandard[i],errorArrayStandard[i],label='standardCombination lmin='+ str(i+minLmin))
        plt.loglog(xFEvalArrayStandard[i],errorArrayStandard[i],label='standardCombination distinct f evals lmin='+ str(i+minLmin))

    for i in range(len(adaptiveAlgorithmVector)):
        print(numNaive[i], errorArray[i], adaptiveAlgorithmVector[i][3] +' Naive evaluation')
        print(numIdeal[i], errorArray[i], adaptiveAlgorithmVector[i][3] +' total points')
        print(numFEvalIdeal[i], errorArray[i], adaptiveAlgorithmVector[i][3] +' distinct f evals')
        plt.loglog(numNaive[i],errorArray[i],label= adaptiveAlgorithmVector[i][3] +' Naive evaluation')
        plt.loglog(numIdeal[i],errorArray[i],label=adaptiveAlgorithmVector[i][3] +' total points')
        plt.loglog(numFEvalIdeal[i],errorArray[i],label=adaptiveAlgorithmVector[i][3] +' distinct f evals')
    plt.legend(bbox_to_anchor=(3,1), loc="upper right")
    plt.xlabel('Number of points')
    plt.ylabel('Approximation error')
    #plt.savefig('convergence.pdf', bbox_inches='tight')
    plt.show()


