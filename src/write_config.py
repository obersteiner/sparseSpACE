import json
import os

class Scenario:

    def __init__(self, parameter, i):
        self.outputPath = "output"
        currentSimulationName = "CampusUtilisation." + str(i)
        self.config = {"name": currentSimulationName, "description": "",
                       "release": "1.0", "commithash": "8c70caee933026d48c0d8e0f1dc23bcb2ed66a4c",
                       "processWriters": {
                           "files": [ {
                               "type": "org.vadere.simulator.projects.dataprocessing.outputfile.TimestepOutputFile",
                               "filename": "timesteps.txt",
                               "processors": [5, 6, 7, 8, 9, 10, 11, 12, 13]
                           }],
                           "processors": [{
                               "type": "org.vadere.simulator.projects.dataprocessing.processor.PedestrianPositionProcessor",
                               "id": 1
                           }, {
                               "type": "org.vadere.simulator.projects.dataprocessing.processor.PedestrianTargetIdProcessor",
                               "id": 2
                           }, {
                               "type": "org.vadere.simulator.projects.dataprocessing.processor.PedestrianOverlapProcessor",
                               "id": 3
                           }, {
                               "type": "org.vadere.simulator.projects.dataprocessing.processor.NumberOverlapsProcessor",
                               "id": 4,
                               "attributesType": "org.vadere.state.attributes.processor.AttributesNumberOverlapsProcessor",
                               "attributes": {
                                   "pedestrianOverlapProcessorId": 3
                               }
                           }, {
                               "type": "org.vadere.simulator.projects.dataprocessing.processor.AreaDensityCountingProcessor",
                               "id": 5,
                               "attributesType": "org.vadere.state.attributes.processor.AttributesAreaDensityCountingProcessor",
                               "attributes": {
                                   "measurementAreaId": 1
                               }
                           }, {
                               "type": "org.vadere.simulator.projects.dataprocessing.processor.AreaDensityCountingProcessor",
                               "id": 6,
                               "attributesType": "org.vadere.state.attributes.processor.AttributesAreaDensityCountingProcessor",
                               "attributes": {
                                   "measurementAreaId": 2
                               }
                           }, {
                               "type": "org.vadere.simulator.projects.dataprocessing.processor.AreaDensityCountingProcessor",
                               "id": 7,
                               "attributesType": "org.vadere.state.attributes.processor.AttributesAreaDensityCountingProcessor",
                               "attributes": {
                                   "measurementAreaId": 3
                               }
                           }, {
                               "type": "org.vadere.simulator.projects.dataprocessing.processor.AreaDensityCountingProcessor",
                               "id": 8,
                               "attributesType": "org.vadere.state.attributes.processor.AttributesAreaDensityCountingProcessor",
                               "attributes": {
                                   "measurementAreaId": 4
                               }
                           }, {
                               "type": "org.vadere.simulator.projects.dataprocessing.processor.AreaDensityCountingProcessor",
                               "id": 9,
                               "attributesType": "org.vadere.state.attributes.processor.AttributesAreaDensityCountingProcessor",
                               "attributes": {
                                   "measurementAreaId": 5
                               }
                           }, {
                               "type": "org.vadere.simulator.projects.dataprocessing.processor.AreaDensityCountingProcessor",
                               "id": 10,
                               "attributesType": "org.vadere.state.attributes.processor.AttributesAreaDensityCountingProcessor",
                               "attributes": {
                                   "measurementAreaId": 6
                               }
                           }, {
                               "type": "org.vadere.simulator.projects.dataprocessing.processor.AreaDensityCountingProcessor",
                               "id": 11,
                               "attributesType": "org.vadere.state.attributes.processor.AttributesAreaDensityCountingProcessor",
                               "attributes": {
                                   "measurementAreaId": 7
                               }
                           }, {
                               "type": "org.vadere.simulator.projects.dataprocessing.processor.AreaDensityCountingProcessor",
                               "id": 12,
                               "attributesType": "org.vadere.state.attributes.processor.AttributesAreaDensityCountingProcessor",
                               "attributes": {
                                   "measurementAreaId": 8
                               }
                           }, {
                               "type": "org.vadere.simulator.projects.dataprocessing.processor.AreaDensityCountingProcessor",
                               "id": 13,
                               "attributesType": "org.vadere.state.attributes.processor.AttributesAreaDensityCountingProcessor",
                               "attributes": {
                                   "measurementAreaId": 9
                               }
                           }],
                           "isTimestamped": False,
                           "isWriteMetaData": False
                       }, "scenario": {
                "mainModel": "org.vadere.simulator.models.osm.OptimalStepsModel",
                "attributesModel": {
                    "org.vadere.state.attributes.models.AttributesOSM": {
                        "stepCircleResolution": 4,
                        "numberOfCircles": 1,
                        "optimizationType": "NELDER_MEAD",
                        "varyStepDirection": True,
                        "movementType": "ARBITRARY",
                        "stepLengthIntercept": 0.4625,
                        "stepLengthSlopeSpeed": 0.2345,
                        "stepLengthSD": 0.036,
                        "movementThreshold": 0.0,
                        "minStepLength": 0.1,
                        "minimumStepLength": True,
                        "maxStepDuration": 1.7976931348623157E308,
                        "dynamicStepLength": True,
                        "updateType": "EVENT_DRIVEN",
                        "seeSmallWalls": False,
                        "targetPotentialModel": "org.vadere.simulator.models.potential.fields.PotentialFieldTargetGrid",
                        "pedestrianPotentialModel": "org.vadere.simulator.models.potential.PotentialFieldPedestrianCompactSoftshell",
                        "obstaclePotentialModel": "org.vadere.simulator.models.potential.PotentialFieldObstacleCompactSoftshell",
                        "submodels": []
                    },
                    "org.vadere.state.attributes.models.AttributesPotentialCompactSoftshell": {
                        "pedPotentialIntimateSpaceWidth": 0.45,
                        "pedPotentialPersonalSpaceWidth": 1.2,
                        "pedPotentialHeight": 50.0,
                        "obstPotentialWidth": 0.8,
                        "obstPotentialHeight": 6.0,
                        "intimateSpaceFactor": 1.2,
                        "personalSpacePower": 1,
                        "intimateSpacePower": 1
                    },
                    "org.vadere.state.attributes.models.AttributesFloorField": {
                        "createMethod": "HIGH_ACCURACY_FAST_MARCHING",
                        "potentialFieldResolution": 0.1,
                        "obstacleGridPenalty": 0.1,
                        "targetAttractionStrength": 1.0,
                        "timeCostAttributes": {
                            "standardDeviation": 0.7,
                            "type": "UNIT",
                            "obstacleDensityWeight": 3.5,
                            "pedestrianSameTargetDensityWeight": 3.5,
                            "pedestrianOtherTargetDensityWeight": 3.5,
                            "pedestrianWeight": 3.5,
                            "queueWidthLoading": 1.0,
                            "pedestrianDynamicWeight": 6.0,
                            "loadingType": "CONSTANT",
                            "width": 0.2,
                            "height": 1.0
                        }
                    }
                },
                "attributesSimulation": {
                    "finishTime": 1000.0,
                    "simTimeStepLength": 0.9,
                    "realTimeSimTimeRatio": 0.0,
                    "writeSimulationData": True,
                    "visualizationEnabled": True,
                    "printFPS": False,
                    "digitsPerCoordinate": 2,
                    "useFixedSeed": True,
                    "fixedSeed": 1,
                    "simulationSeed": 1,
                    "useSalientBehavior": False
                },
                "topography": {
                    "attributes": {
                        "bounds": {
                            "x": 0.0,
                            "y": 0.0,
                            "width": 40.0,
                            "height": 134.0
                        },
                        "boundingBoxWidth": 0.5,
                        "bounded": True
                    },
                    "obstacles": [{
                        "shape": {
                            "x": 0.6,
                            "y": 16.2,
                            "width": 16.449925979160344,
                            "height": 104.1957669874081,
                            "type": "RECTANGLE"
                        },
                        "id": -1
                    }, {
                        "shape": {
                            "x": 0.5,
                            "y": 0.5,
                            "width": 39.02543590126209,
                            "height": 3.2970663939064195,
                            "type": "RECTANGLE"
                        },
                        "id": -1
                    }, {
                        "shape": {
                            "x": 26.994647038738325,
                            "y": 130.020257996712,
                            "width": 12.550090962186982,
                            "height": 3.4797420032880098,
                            "type": "RECTANGLE"
                        },
                        "id": -1
                    }, {
                        "shape": {
                            "x": 30.090567779979768,
                            "y": 76.6259708046802,
                            "width": 9.479438424141012,
                            "height": 53.8263734666745,
                            "type": "RECTANGLE"
                        },
                        "id": -1
                    }, {
                        "shape": {
                            "x": 33.69413796946395,
                            "y": 43.29292653532502,
                            "width": 5.966833766160555,
                            "height": 33.43836499644841,
                            "type": "RECTANGLE"
                        },
                        "id": -1
                    }, {
                        "shape": {
                            "x": 22.053290261983925,
                            "y": 58.67853535353535,
                            "width": 3.6467097380160745,
                            "height": 59.15140379652733,
                            "type": "RECTANGLE"
                        },
                        "id": -1
                    }, {
                        "shape": {
                            "x": 25.663018870408795,
                            "y": 76.65052728543294,
                            "width": 4.449821720659042,
                            "height": 41.17941186462974,
                            "type": "RECTANGLE"
                        },
                        "id": -1
                    }, {
                        "shape": {
                            "x": 22.113330633377807,
                            "y": 3.798564089960138,
                            "width": 6.0209716031147735,
                            "height": 36.79247349147968,
                            "type": "RECTANGLE"
                        },
                        "id": -1
                    }, {
                        "shape": {
                            "x": 28.0958148189324,
                            "y": 3.7769524580379086,
                            "width": 11.438119573128613,
                            "height": 14.322712746030408,
                            "type": "RECTANGLE"
                        },
                        "id": -1
                    }, {
                        "shape": {
                            "x": 31.073455897218633,
                            "y": 29.96973791745603,
                            "width": 8.448327979282244,
                            "height": 13.53026208254397,
                            "type": "RECTANGLE"
                        },
                        "id": -1
                    }, {
                        "shape": {
                            "x": 22.113330633377807,
                            "y": 43.31703354297693,
                            "width": 13.074176331962668,
                            "height": 12.322584304493802,
                            "type": "RECTANGLE"
                        },
                        "id": -1
                    }, {
                        "shape": {
                            "x": 16.597698822828217,
                            "y": 120.0,
                            "width": 10.465528866021305,
                            "height": 0.4000000000000057,
                            "type": "RECTANGLE"
                        },
                        "id": -1
                    }, {
                        "shape": {
                            "x": 27.0,
                            "y": 120.0,
                            "width": 0.3000000000000007,
                            "height": 7.0,
                            "type": "RECTANGLE"
                        },
                        "id": -1
                    }, {
                        "shape": {
                            "x": 31.057840445203936,
                            "y": 19.8,
                            "width": 0.2421595547960642,
                            "height": 10.493819730797295,
                            "type": "RECTANGLE"
                        },
                        "id": -1
                    }, {
                        "shape": {
                            "x": 0.5,
                            "y": 120.4,
                            "width": 16.8,
                            "height": 13.1,
                            "type": "RECTANGLE"
                        },
                        "id": -1
                    }],
                    "measurementAreas": [{
                        "shape": {
                            "x": 25.7,
                            "y": 58.7,
                            "width": 7.900802407221672,
                            "height": 17.97973409869921,
                            "type": "RECTANGLE"
                        },
                        "id": 1
                    }, {
                        "shape": {
                            "x": 0.6,
                            "y": 3.9,
                            "width": 16.414517555004977,
                            "height": 12.199999999999998,
                            "type": "RECTANGLE"
                        },
                        "id": 2
                    }, {
                        "shape": {
                            "x": 17.319337902081617,
                            "y": 120.4,
                            "width": 9.680662097918383,
                            "height": 13.118907280497723,
                            "type": "RECTANGLE"
                        },
                        "id": 3
                    }, {
                        "shape": {
                            "x": 31.3,
                            "y": 18.1,
                            "width": 8.2,
                            "height": 11.869737917456028,
                            "type": "RECTANGLE"
                        },
                        "id": 4
                    }, {
                        "shape": {
                            "x": 17.18,
                            "y": 3.8714411602724397,
                            "width": 4.8,
                            "height": 12.12855883972756,
                            "type": "RECTANGLE"
                        },
                        "id": 6
                    }, {
                        "shape": {
                            "x": 28.30121620442756,
                            "y": 18.3,
                            "width": 2.589454658101552,
                            "height": 1.6421383537386838,
                            "type": "RECTANGLE"
                        },
                        "id": 8
                    }, {
                        "shape": {
                            "x": 25.7,
                            "y": 55.8,
                            "width": 7.9,
                            "height": 2.6,
                            "type": "RECTANGLE"
                        },
                        "id": 5
                    }, {
                        "shape": {
                            "x": 17.18,
                            "y": 16.3,
                            "width": 4.8,
                            "height": 103.61090790033026,
                            "type": "RECTANGLE"
                        },
                        "id": 9
                    }, {
                        "shape": {
                            "x": 27.163887099692758,
                            "y": 127.1,
                            "width": 2.836112900307242,
                            "height": 2.8000000000000114,
                            "type": "RECTANGLE"
                        },
                        "id": 7
                    }],
                    "stairs": [],
                    "targets": [{
                        "id": 1,
                        "absorbing": False,
                        "shape": {
                            "x": 26.39960972333986,
                            "y": 68.3183884988298,
                            "width": 6.647843748185839,
                            "height": 2.5622467024727484,
                            "type": "RECTANGLE"
                        },
                        "waitingTime": 50.0,
                        "waitingTimeYellowPhase": 0.0,
                        "parallelWaiters": 0,
                        "individualWaiting": True,
                        "deletionDistance": 5.0,
                        "startingWithRedLight": False,
                        "nextSpeed": -1.0
                    }, {
                        "id": 2,
                        "absorbing": False,
                        "shape": {
                            "x": 2.996406941532729,
                            "y": 6.318829278105989,
                            "width": 5.134904138896767,
                            "height": 7.160266448552614,
                            "type": "RECTANGLE"
                        },
                        "waitingTime": 50.0,
                        "waitingTimeYellowPhase": 150.0,
                        "parallelWaiters": 0,
                        "individualWaiting": False,
                        "deletionDistance": 5.0,
                        "startingWithRedLight": True,
                        "nextSpeed": -1.0
                    }, {
                        "id": 3,
                        "absorbing": False,
                        "shape": {
                            "x": 17.6870003869054,
                            "y": 128.27072175551422,
                            "width": 2.6299735332790632,
                            "height": 4.545158747630424,
                            "type": "RECTANGLE"
                        },
                        "waitingTime": 50.0,
                        "waitingTimeYellowPhase": 0.0,
                        "parallelWaiters": 0,
                        "individualWaiting": True,
                        "deletionDistance": 3.0,
                        "startingWithRedLight": True,
                        "nextSpeed": -1.0
                    }, {
                        "id": 4,
                        "absorbing": False,
                        "shape": {
                            "x": 34.91835551385038,
                            "y": 25.300605615025137,
                            "width": 3.9642040086986725,
                            "height": 4.002691426258863,
                            "type": "RECTANGLE"
                        },
                        "waitingTime": 100.0,
                        "waitingTimeYellowPhase": 0.0,
                        "parallelWaiters": 0,
                        "individualWaiting": True,
                        "deletionDistance": 5.0,
                        "startingWithRedLight": True,
                        "nextSpeed": -1.0
                    }],
                    "absorbingAreas": [],
                    "sources": [{
                        "id": 1,
                        "shape": {
                            "x": 26.5,
                            "y": 73.4,
                            "width": 1.6,
                            "height": 2.5,
                            "type": "RECTANGLE"
                        },
                        "interSpawnTimeDistribution": "org.vadere.state.scenario.ConstantDistribution",
                        "distributionParameters": [1.0],
                        "spawnNumber": 160,
                        "maxSpawnNumberTotal": -1,
                        "startTime": 0.0,
                        "endTime": 0.0,
                        "spawnAtRandomPositions": False,
                        "useFreeSpaceOnly": True,
                        "targetIds": [2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2,
                                      1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3,
                                      2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1],
                        "groupSizeDistribution": [1.0],
                        "dynamicElementType": "PEDESTRIAN"
                    }, {
                        "id": 2,
                        "shape": {
                            "x": 31.2,
                            "y": 73.36568656117964,
                            "width": 1.4630814813109438,
                            "height": 2.5343134388203623,
                            "type": "RECTANGLE"
                        },
                        "interSpawnTimeDistribution": "org.vadere.state.scenario.ConstantDistribution",
                        "distributionParameters": [1.0],
                        "spawnNumber": 40,
                        "maxSpawnNumberTotal": -1,
                        "startTime": 0.0,
                        "endTime": 0.0,
                        "spawnAtRandomPositions": False,
                        "useFreeSpaceOnly": True,
                        "targetIds": [2, 4, 2, 1, 2, 4, 2, 1, 2, 4, 2, 1, 2, 4, 2, 1, 2, 4, 2, 1, 2, 4, 2, 1, 2, 4, 2,
                                      1, 2, 4, 2, 1, 2, 4, 2, 1, 2, 4, 2, 1, 2, 4, 2, 1, 2, 4, 2, 1, 2, 4, 2, 1, 2, 4,
                                      2, 1, 2, 4, 2, 1, 2, 4, 2, 1, 2, 4, 2, 1, 2, 4, 2, 1, 2, 4, 2, 1, 2, 4, 2, 1],
                        "groupSizeDistribution": [1.0],
                        "dynamicElementType": "PEDESTRIAN"
                    }],
                    "dynamicElements": [],
                    "attributesPedestrian": {
                        "radius": 0.2,
                        "densityDependentSpeed": False,
                        "speedDistributionMean": 1.5,
                        "speedDistributionStandardDeviation": 0.26,
                        "minimumSpeed": 0.5,
                        "maximumSpeed": 2.2,
                        "acceleration": 2.0,
                        "footStepsToStore": 4,
                        "searchRadius": 1.0,
                        "angleCalculationType": "USE_CENTER",
                        "targetOrientationAngleThreshold": 45.0
                    },
                    "teleporter": None,
                    "attributesCar": {
                        "id": -1,
                        "radius": 0.2,
                        "densityDependentSpeed": False,
                        "speedDistributionMean": 1.34,
                        "speedDistributionStandardDeviation": 0.26,
                        "minimumSpeed": 0.5,
                        "maximumSpeed": 2.2,
                        "acceleration": 2.0,
                        "footStepsToStore": 4,
                        "searchRadius": 1.0,
                        "angleCalculationType": "USE_CENTER",
                        "targetOrientationAngleThreshold": 45.0,
                        "length": 4.5,
                        "width": 1.7,
                        "direction": {
                            "x": 1.0,
                            "y": 0.0
                        }
                    }
                },
                "eventInfos": []
            }
                       }
        simulationConfigFile = currentSimulationName + ".scenario"
        currentSimulationPath = self.outputPath + "/" + currentSimulationName
        # if os.path.exists(currentSimulationPath) == False: os.makedirs(currentSimulationPath)
        self.scenariosPath = "scenarios_master"

        #parametersFile = self.outputPath + "/" + currentSimulationName + ".parameters"
        parametersFile=os.getcwd()+"/"+self.outputPath+"/"+currentSimulationName+".parameters"
        with open(parametersFile, 'w') as f:
            f.write(str(parameter) + "\n")

        # parameters
        ippMensaRatio = parameter[0]
        numPersonsInScenario = 200
        numPersonsForIppMensa = int(numPersonsInScenario * ippMensaRatio)
        numPersonsForTumMensa = numPersonsInScenario - numPersonsForIppMensa

        residenceTimeIpp = parameter[1]
        residenceTimeMensa = parameter[2]
        meanSpeed = parameter[3]
        # simTimeStepLength = parameter[4]

        self.config['scenario']['topography']['sources'][0]['spawnNumber'] = float(numPersonsForTumMensa)
        self.config['scenario']['topography']['sources'][1]['spawnNumber'] = float(numPersonsForIppMensa)

        self.config['scenario']['topography']['targets'][3]['waitingTime'] = float(residenceTimeIpp)
        self.config['scenario']['topography']['targets'][2]['waitingTime'] = float(residenceTimeMensa)

        self.config['scenario']['topography']['attributesPedestrian']['speedDistributionMean'] = float(meanSpeed)
        # self.config['vadere']['attributesSimulation']['randomSeed'] = seed
        # self.config['vadere']['attributesSimulation']['simTimeStepLength'] = simTimeStepLength

        # dump
    def printScenario(self, currentSimulationName):
        json.dump(self.config, open(self.scenariosPath + "/" + currentSimulationName+".scenario", 'w'), indent=2)
         