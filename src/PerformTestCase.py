from DimAdaptiveCombi import *


def performTestStandard(f, a, b, grid, lmin, maxLmax, dim, reference_solution, evaluation_points):
    # calculate standard combination scheme results
    errorArrayStandard = []
    pointArray = []
    distinctFEvalArray = []
    operation = Integration(f, grid, dim, reference_solution)
    standardCombi = StandardCombi(a, b, operation=operation)
    interpolation_errorL2 = []
    interpolation_errorMax = []
    for i in range(lmin + 1, lmin + maxLmax):
        scheme, error, result = standardCombi.perform_operation(lmin, i)
        errorArrayStandard.append(error / abs(reference_solution))
        pointArray.append(standardCombi.get_total_num_points())
        distinctFEvalArray.append(standardCombi.get_total_num_points(distinct_function_evals=True))
        if evaluation_points is not None:
            interpolated_values = np.asarray(standardCombi(evaluation_points))
            real_values = np.asarray([f.eval(point) for point in evaluation_points])
            diff = [real_values[i] - interpolated_values[i] for i in range(len(evaluation_points))]
            # print(interpolated_values, diff)
            interpolation_errorL2.append(scipy.linalg.norm(diff, 2))
            interpolation_errorMax.append(scipy.linalg.norm(diff, np.inf))
    return pointArray, distinctFEvalArray, errorArrayStandard, interpolation_errorL2, interpolation_errorMax


def performTestcaseArbitraryDim(f, a, b, adaptiveAlgorithmVector, maxtol, dim, maxLmax, grid=None, minLmin=1, maxLmin=3,
                                minTol=-1, doDimAdaptive=False, max_evaluations=10**7, evaluation_points=None):
    # realIntegral = scipy.integrate.dblquad(f, a, b, lambda x:a, lambda x:b, epsabs=1e-15, epsrel=1e-15)[0]
    reference_solution = f.getAnalyticSolutionIntegral(a, b)
    print("Exact integral", reference_solution)
    errorArray = []
    surplusErrorArray = []

    numEvaluationsArray = []
    numNaive = []
    numIdeal = []
    numFEvalIdeal = []
    interpolation_error_arrayL2 = []
    interpolation_error_arrayMax = []

    # calculate refinements for different tolerance values
    for algorithm in adaptiveAlgorithmVector:
        errorArrayAlgorithm = []
        surplusErrorArrayAlgorithm = []
        numEvaluationsArrayAlgorithm = []
        numNaiveAlgorithm = []
        numIdealAlgorithm = []
        numFEvalIdealAlgorithm = []
        '''
        for i in range(minTol, maxtol+1):
            start = time.time()
            if i == minTol:
                coarsening, combischeme, lmax, integral, numberOfEvaluations, error_array_new, num_point_array_new = algorithm[0].performSpatiallyAdaptiv(
                    algorithm[1], algorithm[2], f, algorithm[3], 10 ** -i, reference_solution=reference_solution)
                # errorArrayAlgorithm.append(abs(integral - realIntegral) / abs(realIntegral))
                errorArrayAlgorithm.extend(error_array_new)
                numEvaluationsArrayAlgorithm.append(numberOfEvaluations)
                # numIdealAlgorithm.extend(algorithm[0].get_total_num_points_arbitrary_dim(False))
                # numNaiveAlgorithm.append(algorithm[0].get_total_num_points_arbitrary_dim(True))
                numFEvalIdealAlgorithm.extend(num_point_array_new)
            else:
                if abs(integral - reference_solution) / abs(reference_solution) > 10 ** -i:
                    coarsening, combischeme, lmax, integral, numberOfEvaluations, error_array_new, num_point_array_new = algorithm[0].performSpatiallyAdaptiv(
                        algorithm[1], algorithm[2], f, algorithm[3], 10 ** -i, coarsening, reference_solution=reference_solution)
                    #errorArrayAlgorithm.append(abs(integral - realIntegral) / abs(realIntegral))
                    errorArrayAlgorithm.extend(error_array_new)
                    numEvaluationsArrayAlgorithm.append(numberOfEvaluations)
                    #numIdealAlgorithm.extend(algorithm[0].get_total_num_points_arbitrary_dim(False))
                    #numNaiveAlgorithm.append(algorithm[0].get_total_num_points_arbitrary_dim(True))
                    numFEvalIdealAlgorithm.extend(num_point_array_new)
            end = time.time()
            print("time spent in case", i, end - start)
            '''
        coarsening, combischeme, lmax, integral, numberOfEvaluations, error_array_new, num_point_array_new, surplus_error_array_new, interpolation_errorL2, interpolation_errorMax = \
        algorithm[
            0].performSpatiallyAdaptiv(
            algorithm[1], algorithm[2], algorithm[3], 10 ** -maxtol, max_evaluations=max_evaluations, evaluation_points=evaluation_points)
        # errorArrayAlgorithm.append(abs(integral - realIntegral) / abs(realIntegral))
        errorArrayAlgorithm.extend(error_array_new)
        surplusErrorArrayAlgorithm.extend(surplus_error_array_new)

        numEvaluationsArrayAlgorithm.append(numberOfEvaluations)
        # numIdealAlgorithm.extend(algorithm[0].get_total_num_points_arbitrary_dim(False))
        # numNaiveAlgorithm.append(algorithm[0].get_total_num_points_arbitrary_dim(True))
        numFEvalIdealAlgorithm.extend(num_point_array_new)

        errorArray.append(errorArrayAlgorithm)
        surplusErrorArray.append(surplusErrorArrayAlgorithm)

        numEvaluationsArray.append(numEvaluationsArrayAlgorithm)
        numNaive.append(numNaiveAlgorithm)
        numIdeal.append(numIdealAlgorithm)
        numFEvalIdeal.append(numFEvalIdealAlgorithm)
        interpolation_error_arrayL2.append(interpolation_errorL2)
        interpolation_error_arrayMax.append(interpolation_errorMax)



    if doDimAdaptive:
        dimAdaptiveCombi = DimAdaptiveCombi(a, b, grid)
        scheme, error, result, errorArrayDimAdaptive, numFEvalIdealDimAdaptive = dimAdaptiveCombi.perform_combi(1, 2, f,
                                                                                                                10 ** -maxtol,
                                                                                                                reference_solution=reference_solution)

    # calculate different standard combination scheme results
    xArrayStandard = []
    xFEvalArrayStandard = []
    errorArrayStandard = []
    interpolation_error_standardL2 = []
    interpolation_error_standardMax = []

    for i in range(minLmin, maxLmin):
        xArrayStandardTest, xFEvalArrayStandardTest, errorArrayStandardTest, interpolation_errorL2, interpolation_errorMax = performTestStandard(f, a, b, grid, i,
                                                                                                  maxLmax - (i - 1) * (
                                                                                                          dim - 1) + i - 1,
                                                                                                  dim,
                                                                                                  reference_solution, evaluation_points=evaluation_points)
        xArrayStandard.append(xArrayStandardTest)
        xFEvalArrayStandard.append(xFEvalArrayStandardTest)
        errorArrayStandard.append(errorArrayStandardTest)
        interpolation_error_standardL2.append(interpolation_errorL2)
        interpolation_error_standardMax.append(interpolation_errorMax)
    # plot
    for i in range(maxLmin - minLmin):
        print(xArrayStandard[i], errorArrayStandard[i], "Number of Points Standard lmin= " + str(i + minLmin))
        print(xFEvalArrayStandard[i], errorArrayStandard[i], "Distinct f evaks Standard lmin= " + str(i + minLmin))
        print(xFEvalArrayStandard[i], interpolation_error_standardL2[i],  "L2 interpolation error lmin= " + str(i + minLmin))
        print(xFEvalArrayStandard[i], interpolation_error_standardMax[i], "Linf interpolation error lmin= " + str(i + minLmin))

        # plt.loglog(xArrayStandard[i],errorArrayStandard[i],label='standardCombination lmin='+ str(i+minLmin))
        #plt.loglog(xFEvalArrayStandard[i], errorArrayStandard[i],
        #           label='standardCombination distinct f evals lmin=' + str(i + minLmin))

        plt.loglog(xFEvalArrayStandard[i], interpolation_error_standardL2[i],
                   label='standardCombination L2 lmin=' + str(i + minLmin))
        plt.loglog(xFEvalArrayStandard[i], interpolation_error_standardMax[i],
                   label='standardCombination Linf lmin=' + str(i + minLmin))
    if doDimAdaptive:
        print("numPoints =", numFEvalIdealDimAdaptive)
        print("error=", errorArrayDimAdaptive, "Number of Points DimAdaptive lmin= 1")
        plt.loglog(numFEvalIdealDimAdaptive, errorArrayDimAdaptive, label="Number of Points DimAdaptive lmin= 1")
    for i in range(len(adaptiveAlgorithmVector)):
        # print(numNaive[i], errorArray[i], adaptiveAlgorithmVector[i][4] + ' Naive evaluation')
        # print(numIdeal[i], errorArray[i], adaptiveAlgorithmVector[i][4] + ' total points')
        print(numFEvalIdeal[i], errorArray[i], adaptiveAlgorithmVector[i][4] + ' distinct f evals')
        print(numFEvalIdeal[i], surplusErrorArray[i], adaptiveAlgorithmVector[i][4] + ' surplus error distinct f evals')
        print(numFEvalIdeal[i],interpolation_error_arrayL2[i], adaptiveAlgorithmVector[i][4] + ' L2 interpolation error')
        print(numFEvalIdeal[i], interpolation_error_arrayMax[i], adaptiveAlgorithmVector[i][4] + ' Linf interpolation error')

        # plt.loglog(numNaive[i],errorArray[i],label= adaptiveAlgorithmVector[i][3] +' Naive evaluation')
        # plt.loglog(numIdeal[i],errorArray[i],label=adaptiveAlgorithmVector[i][3] +' total points')
        #plt.loglog(numFEvalIdeal[i], errorArray[i], label=adaptiveAlgorithmVector[i][4] + ' distinct f evals')
        plt.loglog(numFEvalIdeal[i], interpolation_error_arrayL2[i], label=adaptiveAlgorithmVector[i][4] + ' L2')
        plt.loglog(numFEvalIdeal[i], interpolation_error_arrayMax[i], label=adaptiveAlgorithmVector[i][4] + ' Linf')
        plt.loglog(numFEvalIdeal[i], surplusErrorArray[i], '--', label=adaptiveAlgorithmVector[i][4] + ' surplus error')

    plt.legend(bbox_to_anchor=(3, 1), loc="upper right")
    plt.xlabel('Number of points')
    plt.ylabel('Approximation error')
    # plt.savefig('convergence.pdf', bbox_inches='tight')
    plt.show()
