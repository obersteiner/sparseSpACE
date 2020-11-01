from DimAdaptiveCombi import *
import matplotlib.cm as mplcm
import matplotlib, numpy as np, matplotlib.pyplot as plt


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


def performTestcaseArbitraryDim(f, a, b, adaptiveAlgorithmVector, maxtol, dim, maxLmax, grids=None,
                                minLmin=1, maxLmin=3,
                                minTol=-1, doDimAdaptive=False, max_evaluations=10 ** 7, evaluation_points=None,
                                calc_standard_schemes=True, grid_names=None,
                                legend_title="",
                                filepath="./Results/", filename=None, save_plot:bool=False, save_csv:bool=False, clear_csv:bool=False):
    # realIntegral = scipy.integrate.dblquad(f, a, b, lambda x:a, lambda x:b, epsabs=1e-15, epsrel=1e-15)[0]
    reference_solution = f.getAnalyticSolutionIntegral(a, b)
    print("Exact integral", reference_solution)
    errorArray = []
    surplusErrorArray = []

    numEvaluationsArray = []
    numNaive = []
    numIdeal = []
    numFEvalIdeal = []
    # interpolation_error_arrayL2 = []
    # interpolation_error_arrayMax = []

    # https://stackoverflow.com/questions/8389636/creating-over-20-unique-legend-colors-using-matplotlib
    NUM_COLORS = len(adaptiveAlgorithmVector) + (maxLmin - minLmin)

    # cm = plt.get_cmap('tab20')  # alternative: gist_rainbow, hsv, tab20
    # cNorm = colors.Normalize(vmin=0, vmax=NUM_COLORS - 1)
    # scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='w')
    ax = fig.add_subplot(111)
    # ax.set_prop_cycle('color', [scalarMap.to_rgba((4 * i + 2) % NUM_COLORS) for i in range(NUM_COLORS)])
    from_list = matplotlib.colors.LinearSegmentedColormap.from_list
    cm = from_list('Set1', plt.cm.Set1(range(0, NUM_COLORS)), NUM_COLORS)
    # plt.set_cmap(cm)

    # Clear CSV
    if save_csv and clear_csv:
        clear_csv(filepath, filename)

    print("\n\n\n\n\n\n -------------------------- Start {} -------------------------- \n\n\n\n\n\n".format(legend_title))

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
                algorithm[1], algorithm[2], algorithm[3], 10 ** -maxtol, max_evaluations=max_evaluations,
                evaluation_points=evaluation_points)
        # errorArrayAlgorithm.append(abs(integral - reference_solution) / abs(reference_solution))
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
        # interpolation_error_arrayL2.append(interpolation_errorL2)
        # interpolation_error_arrayMax.append(interpolation_errorMax)

        # Export Data to csv
        if save_csv:
            export_to_csv(filepath, filename, [(algorithm[4], numFEvalIdealAlgorithm, errorArrayAlgorithm)])

        # Spacing in console
        print("\n\n\n\n\n\n-----------------------------------------------------------------------------------------------\n\n\n\n\n\n")

    errorArrayDimAdaptiveArray = []
    numFEvalIdealDimAdaptiveArray = []

    if doDimAdaptive:
        for i, grid in enumerate(grids):
            dimAdaptiveCombi = DimAdaptiveCombi(a, b, grid)
            operation = Integration(f, grid, dim, reference_solution)
            dimAdaptiveCombi = DimAdaptiveCombi(a, b, operation=operation)
            scheme, error, result, errorArrayDimAdaptive, numFEvalIdealDimAdaptive = dimAdaptiveCombi.perform_combi(1,
                                                                                                                    1,
                                                                                                                    10 ** -maxtol,
                                                                                                                    max_number_of_points=max_evaluations)
            errorArrayDimAdaptiveArray.append(errorArrayDimAdaptive)
            numFEvalIdealDimAdaptiveArray.append(numFEvalIdealDimAdaptive)

            # Export Data to csv
            if save_csv:
                name = "{}: Number of Points DimAdaptive lmin= 1".format(grid_names[i])
                export_to_csv(filepath, filename, [(name, numFEvalIdealDimAdaptive, errorArrayDimAdaptive)])

    # calculate different standard combination scheme results
    if calc_standard_schemes:
        for k, grid in enumerate(grids):
            xArrayStandard = []
            xFEvalArrayStandard = []
            errorArrayStandard = []
            # interpolation_error_standardL2 = []
            # interpolation_error_standardMax = []

            for i in range(minLmin, maxLmin):
                xArrayStandardTest, \
                xFEvalArrayStandardTest, \
                errorArrayStandardTest, \
                interpolation_errorL2, \
                interpolation_errorMax = performTestStandard(f, a, b, grid, i,
                                                             maxLmax - (i - 1) * (
                                                                     dim - 1) + i - 1,
                                                             dim,
                                                             reference_solution, evaluation_points=evaluation_points)

                xArrayStandard.append(xArrayStandardTest)
                xFEvalArrayStandard.append(xFEvalArrayStandardTest)
                errorArrayStandard.append(errorArrayStandardTest)
                # interpolation_error_standardL2.append(interpolation_errorL2)
                # interpolation_error_standardMax.append(interpolation_errorMax)

            # plot
            print(maxLmin)
            print(minLmin)
            for i in range(maxLmin - minLmin):
                print(xArrayStandard[i], errorArrayStandard[i], "Number of Points Standard lmin= " + str(i + minLmin))
                print(xFEvalArrayStandard[i], errorArrayStandard[i], "Distinct f evals Standard lmin= " + str(i + minLmin))
                # print(xFEvalArrayStandard[i], interpolation_error_standardL2[i],  "L2 interpolation error lmin= " + str(i + minLmin))
                # print(xFEvalArrayStandard[i], interpolation_error_standardMax[i], "Linf interpolation error lmin= " + str(i + minLmin))

                # ax.loglog(xArrayStandard[i], errorArrayStandard[i], label='standardCombination lmin=' + str(i + minLmin))
                name = "{} (Standard Combi) lmin={}".format(grid_names[k], str(i + minLmin))
                ax.loglog(xFEvalArrayStandard[i], errorArrayStandard[i], "--", label=name + " distinct f evals")

                # Export Data to csv
                if save_csv:
                    export_to_csv(filepath, filename, [(name, xFEvalArrayStandard[i], errorArrayStandard[i])])

                # ax.loglog(xFEvalArrayStandard[i], interpolation_error_standardL2[i],
                #            label='standardCombination L2 lmin=' + str(i + minLmin))
                # ax.loglog(xFEvalArrayStandard[i], interpolation_error_standardMax[i],
                #            label='standardCombination Linf lmin=' + str(i + minLmin))

    if doDimAdaptive:
        for i in range(len(numFEvalIdealDimAdaptiveArray)):
            name = "{}: Number of Points DimAdaptive lmin= 1".format(grid_names[i])
            print("numPoints =", numFEvalIdealDimAdaptiveArray[i])
            print("error=", errorArrayDimAdaptiveArray[i], name)
            ax.loglog(numFEvalIdealDimAdaptiveArray[i], errorArrayDimAdaptiveArray[i], label=name)

    line = '-'

    for i in range(len(adaptiveAlgorithmVector)):
        # if line == '-':
        #     line = '--'
        # elif line == '--':
        #     line = '-'

        # print(numNaive[i], errorArray[i], adaptiveAlgorithmVector[i][4] + ' Naive evaluation')
        # print(numIdeal[i], errorArray[i], adaptiveAlgorithmVector[i][4] + ' total points')
        print(numFEvalIdeal[i], errorArray[i], adaptiveAlgorithmVector[i][4] + ' error (distinct f evals)')
        # print(numFEvalIdeal[i], surplusErrorArray[i], adaptiveAlgorithmVector[i][4] + ' surplus error distinct f evals')

        # print(numFEvalIdeal[i],interpolation_error_arrayL2[i], adaptiveAlgorithmVector[i][4] + ' L2 interpolation error')
        # print(numFEvalIdeal[i], interpolation_error_arrayMax[i], adaptiveAlgorithmVector[i][4] + ' Linf interpolation error')

        # ax.loglog(numNaive[i],errorArray[i],label= adaptiveAlgorithmVector[i][3] +' Naive evaluation')
        # ax.loglog(numIdeal[i],errorArray[i],label=adaptiveAlgorithmVector[i][3] +' total points')
        name = adaptiveAlgorithmVector[i][4]
        ax.loglog(numFEvalIdeal[i], errorArray[i], line, label=name + ' error (distinct f evals)')

        # ax.loglog(numFEvalIdeal[i], interpolation_error_arrayL2[i], label=adaptiveAlgorithmVector[i][4] + ' L2')
        # ax.loglog(numFEvalIdeal[i], interpolation_error_arrayMax[i], label=adaptiveAlgorithmVector[i][4] + ' Linf')

        # ax.loglog(numFEvalIdeal[i], surplusErrorArray[i], '--', label=adaptiveAlgorithmVector[i][4] + ' surplus error')

    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=1, mode="expand", borderaxespad=0., title=legend_title)
    ax.set_xlabel('Number of points')
    ax.set_ylabel('Approximation error')

    if save_plot:
        ax.figure.savefig("{}{}.pdf".format(filepath, filename), bbox_inches='tight', dpi=300)

    # ax.figure.show()

def clear_csv(filepath, filename):
    file = "{}{}.csv".format(filepath, filename)
    print("Clearing {}".format(file))

    f = open(file, "w+")
    f.close()


def export_to_csv(filepath, filename, export_data):
    with open("{}{}.csv".format(filepath, filename), 'a') as out:
        csv_out = csv.writer(out, delimiter="|", quoting=csv.QUOTE_NONNUMERIC)

        for name, num_points, error in export_data:
            csv_out.writerow(["name", name])
            csv_out.writerow(["num_points"] + num_points)
            csv_out.writerow(["error"] + error)
            csv_out.writerow([])
