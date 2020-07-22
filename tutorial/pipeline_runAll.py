# This script runs all the pipeline steps
import pipeline_commons
import sys
import os
import time
import argparse
import tensorflow as tf
import pipeline_params
import pipeline_extractRGBImages
import pipeline_segmentation
import pipeline_pointCloudReconstruction
import pipeline_extractPeopleAndCars
import pipeline_3DReconstruction
import pipeline_outputProcessing
import pandas as pd
import copy
import ray
import torch
import torch.multiprocessing as mp
from typing import List, Tuple, Dict, Set
import glob
import re

import logging
from pipeline_commons import globalLogger
import ReconstructionUtils
from PIL import Image

from enum  import Enum

class PipelineStages(Enum):
    PIPELINE_RGB = 0,
    PIPELINE_SEMANTIC = 1,
    PIPELINE_EXTRACTION = 2,
    PIPELINE_POINTCLOUD = 3,
    PIPELINE_3DRECONSTRUCTION = 4,
    PIPELINE_OUTPUT = 5

MIN_DENSITY_TO_USE_SCENE = 10.0
COMPLETED_CHECKPOINT_PREFIXNAME = "completed_"
SEGMENTATION_LAYER_ID = "2"


def parseArguments():
    parser = argparse.ArgumentParser(description="Pipeline process for extracting Waymo data")
    parser.add_argument("--fixDataset", required=False, type=int, default=0,
                        help="Put on 1 if dataset generated so far has some issues...")
    parser.add_argument("--cleanOutputPath", required=True, type=str,
                        help="Folder path to output the extraction only if really needed files")
    parser.add_argument("--fullOutputPath", required=True, type=str,
                        help="Folder path to output the extraction of all intermeddiate files too")
    parser.add_argument("--scenesFile", required=True, type=str,
                        help="If data is already in the output folder, should I recompute")
    parser.add_argument("--DEBUG_MIN_FRAME", required=False, type=int,
                        help="Frame index to start extraction from, 0 is default", default=0)
    parser.add_argument("--DEBUG_MAX_FRAME", required=False, type=int,
                        help="Frame index to end extraction from", default=999999)
    parser.add_argument("--forceRecompute", required=True, type=int,
                        help="If data is already in the output folder, should I recompute")
    parser.add_argument("--forcedSegmentName", required=False, type=str,  help="force running only this segment by its name", default="")
    parser.add_argument("--outputFullCloudReconstruction", required=False, type=int, default=0, help ="if true, cars and pedestrians will be added to point cloud reconstruction")
    parser.add_argument("--scaleUsedInFinalReconstruction", required=False, type=float, default=1.0, help="The scape up/down value (base is 1.0) used to recompose the final ply scenes from frames ply files")
    parser.add_argument("--pointCloudForMotionFrames", required=False, type=int, default=0, help ="if true, a separate ply file will be created for each frame with motions and in the end combined on environment to have motion on each frame too !")
    parser.add_argument("--noiseFilteringWithKNN", required=False, type=int, default=0, help ="if true, a KNN algorithm to remove noise and votes for most common label is used")
    parser.add_argument("--noiseFilteringWithKNNStatistical", required=False, type=int, default=0, help ="if true, a KNN algorithm - fast implemented in CPP + statistical data gathering to remove noise and votes for most common label is used")
    parser.add_argument("--noiseFilteringWithVoxelization", required=False, type=int, default=0, help ="if true, a voxelization algorithm to remove noise and votes for most common label is used")
    parser.add_argument("--keepOriginalFloatingPoints", required=False, type=int, default=0, help="Otherwise we do voxelization which will lower the quality but increase speed. This should be used mostly for rendering / vis purposes")
    parser.add_argument("--overridePedAndCarsBboxExtension", required=False, type=float, default=1.25, help="How much (percent) to extend the bboxes of pedestrians and cars in computations")
    parser.add_argument("--USE_MULTIPROCESSING", required=True, type=int,
                        help="Should we use actors parallelization ?")
    parser.add_argument("--NUM_ACTORS_CPU", required=True, type=int,
                        help="Number of CPU actors in parallelization ?")
    parser.add_argument("--NUM_ACTORS_GPU", required=True, type=int,
                        help="Number of GPU actors in parallelization ?")
    parser.add_argument("--LAYERS_TO_RUN", required=True, type=str,
                        help="Integers split by comma representing which layers indices to run")

    parser.add_argument("--fixIndexStart", required=False, type=int, default = 0,
                        help="Where to start fixing from")

    parser.add_argument("--fixIndexEnd", required=False, type=int, default = -1,
                        help="Where to stop fixing")

    parser.add_argument("--batchSize", required=True, type=int, default=10, help="Size of scene batches")
    parser.add_argument("--indexSTART", required=True, type=int, default=0, help="Index from scenes to start processing")
    parser.add_argument("--indexEND", required=True, type=int, default=0, help="Index from scenes to end processing")

    aargs = parser.parse_args(sys.argv[1:])

    # Check args validity
    if aargs.fixDataset == 1:
        assert aargs.fixIndexStart >= 0 and aargs.fixIndexStart <= aargs.fixIndexEnd, "INvalid fix ranges for scenes !"

    return aargs

def isSegmentationTheOnlyStage(stages):
    return len(stages) == 1 and PipelineStages.PIPELINE_SEMANTIC in stages

def isSegmentationTheOnlyLayer(layers):
    return len(layers) == 1 and SEGMENTATION_LAYER_ID in layers

"""
def dumyWorkFunc(task):
    print("Completing task ", task)
    torch.cuda.set_device(task)
    pass
"""

def workFunc(workData):
    allTasksInBatch = workData[0]
    indicesForThisWorker = workData[1]
    params = workData[2]
    stagesToRun = workData[3]
    id = workData[4]
    globalLogger.info(f"Id {id} got task to solve indices {indicesForThisWorker}")

    forceRecompute = params.FORCE_RECOMPUTE

    def writeCheckpoint(filenames : str, checkpointType : PipelineStages):
        with open(filenames[checkpointType], "w") as outputCheckpointHandle:
            #outputCheckpointHandle.write("")
            pass

    def hasCheckpoint(filenames : str, checkpointType : PipelineStages):
        return os.path.exists(filenames[checkpointType])

    for segment_index in indicesForThisWorker:
        # print("Line {}: {}".format(cnt, segmentPath.strip()))
        segmentPath = allTasksInBatch[segment_index]
        globalLogger.info(f"$$$$$$$$$$$$ Id {id} Processing segment index: {segment_index}, path: {segmentPath} $$$$$$$$$$$$")

        segmentName = pipeline_commons.extractSegmentNameFromPath(segmentPath)
        outputCheckpointFilePrefix = os.path.join(params.BASE_OUTPUT_PATH, f"{COMPLETED_CHECKPOINT_PREFIXNAME}{segmentName}_")
        outputCheckpointFileNames = {}
        for item in PipelineStages:
            outputCheckpointFileNames[item] = outputCheckpointFilePrefix + str(item.name)

        start_time = time.time()
        if not isSegmentationTheOnlyStage(stagesToRun):
            #print("Finding num frames for ", stagesToRun)
            numFrames = pipeline_commons.getNumFramesInSegmentPath(segmentPath)
            params.FRAMEINDEX_MAX = min(params.FRAMEINDEX_MAX, numFrames - 1)
            params.FRAMEINDEX_MIN = max(0, params.FRAMEINDEX_MIN)
            globalLogger.info( f"Id {id} In segment index {segment_index} - There are {numFrames} frames. Starting the pipeline process between frames {pipeline_params.globalParams.FRAMEINDEX_MIN}-{pipeline_params.globalParams.FRAMEINDEX_MAX}, recompute: {pipeline_params.globalParams.FORCE_RECOMPUTE}")
        else:
            globalLogger.info( f"Id {id} Starting the pipeline process between frames {pipeline_params.globalParams.FRAMEINDEX_MIN}-{pipeline_params.globalParams.FRAMEINDEX_MAX}, recompute: {pipeline_params.globalParams.FORCE_RECOMPUTE}")


        hackEnabled = False
        if hackEnabled:
            writeCheckpoint(outputCheckpointFileNames, PipelineStages.PIPELINE_RGB)

        if PipelineStages.PIPELINE_RGB in stagesToRun:
            if forceRecompute == True or not hasCheckpoint(outputCheckpointFileNames, PipelineStages.PIPELINE_RGB):
                globalLogger.log(logging.INFO, f"Id {id} ##Stage 1: RGB Image extraction")
                pipeline_extractRGBImages.do_RGBExtraction(segmentPath, params)
                writeCheckpoint(outputCheckpointFileNames, PipelineStages.PIPELINE_RGB)
            else:
                globalLogger.log(logging.INFO, f"Id {id} ##Stage 1: Already done !")

        if PipelineStages.PIPELINE_SEMANTIC in stagesToRun:
            if forceRecompute == True or not hasCheckpoint(outputCheckpointFileNames, PipelineStages.PIPELINE_SEMANTIC):
                globalLogger.log(logging.INFO, f"Id {id} ##Stage 2: Segmentation")
                pipeline_segmentation.runSegmentationOps(segmentPath, params)
                writeCheckpoint(outputCheckpointFileNames, PipelineStages.PIPELINE_SEMANTIC)
            else:
                globalLogger.log(logging.INFO, f"Id {id} ##Stage 2: Already done !")

        # time.sleep(10)
        if PipelineStages.PIPELINE_EXTRACTION in stagesToRun:
            if forceRecompute == True or not hasCheckpoint(outputCheckpointFileNames, PipelineStages.PIPELINE_EXTRACTION):
                globalLogger.log(logging.INFO, f"Id {id} ##Stage 3: Peoples and vehicle trajectories extraction")
                pipeline_extractPeopleAndCars.do_PeopleAndVehiclesExtraction(segmentPath, params)
                writeCheckpoint(outputCheckpointFileNames, PipelineStages.PIPELINE_EXTRACTION)
            else:
                globalLogger.log(logging.INFO, f"Id {id} ##Stage 3: Already done !")

        if PipelineStages.PIPELINE_POINTCLOUD in stagesToRun:
            if forceRecompute == True or not hasCheckpoint(outputCheckpointFileNames, PipelineStages.PIPELINE_POINTCLOUD):
                globalLogger.log(logging.INFO, f"Id {id} ##Stage 4: Point cloud reconstruction. Part 1: for environment")
                pipeline_pointCloudReconstruction.do_PointCloudReconstruction(segmentPath, params, useEnvironmentPoints=True)

                if params.POINT_CLOUD_FOR_MOTION_FRAMES:
                    globalLogger.log(logging.INFO, f"Id {id} ##Stage 4: Point cloud reconstruction. Part 2: for motion stuff")
                    pipeline_pointCloudReconstruction.do_PointCloudReconstruction(segmentPath, params, useEnvironmentPoints=False)

                writeCheckpoint(outputCheckpointFileNames, PipelineStages.PIPELINE_POINTCLOUD)
            else:
                globalLogger.log(logging.INFO, f"Id {id} ##Stage 4: Already done !")

        if PipelineStages.PIPELINE_3DRECONSTRUCTION in stagesToRun:
            if forceRecompute == True or not hasCheckpoint(outputCheckpointFileNames, PipelineStages.PIPELINE_3DRECONSTRUCTION):
                globalLogger.log(logging.INFO, f"Id {id} ##Stage 5: 3D Scene reconstruction")
                pipeline_3DReconstruction.do_3DReconstruction(segmentPath, params)
                writeCheckpoint(outputCheckpointFileNames, PipelineStages.PIPELINE_3DRECONSTRUCTION)
            else:
                globalLogger.log(logging.INFO, f"Id {id} ##Stage 5: Already done !")

        if PipelineStages.PIPELINE_OUTPUT in stagesToRun:
            if forceRecompute == True or not hasCheckpoint(outputCheckpointFileNames, PipelineStages.PIPELINE_OUTPUT):
                globalLogger.log(logging.INFO, f"Id {id} ##Stage 6: Copy files to output")
                pipeline_outputProcessing.do_output(segmentPath, params)
                writeCheckpoint(outputCheckpointFileNames, PipelineStages.PIPELINE_OUTPUT)
            else:
                globalLogger.log(logging.INFO, f"Id {id} ##Stage 6: Already done !")

        end_time = time.time()
        globalLogger.log(logging.INFO, f"Id {id} $$$ Processing of index {segment_index} finished, time spent {(end_time - start_time)}")
    return 0

# Cyclic append
def getTaskAssignment(batchIndices, numWorkers):
    outRes = [None]*numWorkers
    for i in range(numWorkers):
        outRes[i] = []
    for i in batchIndices:
        outRes[(i % numWorkers)].append(i)
    return outRes

@ray.remote
class Simulator(object):
    def __init__(self, workData):
        self.workData = workData

    def simulate(self):
        with tf.device('/CPU'):
            workFunc(self.workData)
        return 0

def getCompletedCheckpointsForType(basePath: str, type : PipelineStages):
    # TODO for more types...
    completedFiles_RGB = glob.glob(basePath + "/completed_[0-9_A-Za-z]*_RGB")
    completedSegmentNames_RGB = [(R.split("/")[-1][len(COMPLETED_CHECKPOINT_PREFIXNAME):][:-len("_PIPELINE_RGB")]) for R in completedFiles_RGB]
    #completedSegmentNames_RGB = [os.path.join("".join(R.split("/")[:-1]), R]) for R in completedFiles_RGB]
    return completedSegmentNames_RGB

# Returns a tuple containing: the dictionary of segments and frames where we have broken data , AND a set of indices that are processed yet inside the range
def verifyRGBExtracted(allArgs, params, allSortedScenesPaths) -> Tuple[Dict[str, List[int]], Set[int]]:

    # Get the scene paths interested in
    scenePathsSlice = allSortedScenesPaths[allArgs.fixIndexStart : (allArgs.fixIndexEnd + 1)]
    setOfExpectedNonSolvedIndices : Set[int]= set()
    for i in range(allArgs.fixIndexStart, allArgs.fixIndexEnd + 1):
        setOfExpectedNonSolvedIndices.add(i)

    # Get all the completed RGB segments
    outDict = {}
    completedSegmentNames = getCompletedCheckpointsForType(params.SEG_INPUT_IMAGES_BASEPATH, PipelineStages.PIPELINE_RGB)
    numFoldersChecked = 0
    for segmentName in completedSegmentNames:
        rgbFolder = os.path.join(params.SEG_INPUT_IMAGES_BASEPATH, segmentName, params.SEG_INPUT_IMAGES_RGBFOLDER)

        # Check the index of this segment in the input dataset entries
        foundIndex = -1
        for index, sceneFullPath in enumerate(scenePathsSlice):
            if segmentName in sceneFullPath:
                foundIndex = index + allArgs.fixIndexStart
                break
        if foundIndex == -1:
            continue #, f"I couldn't find this folder {segmentName} in the dataset at all !!"

        #if not (allArgs.fixIndexStart <= foundIndex and foundIndex <= allArgs.fixIndexEnd):
        #    continue

        setOfExpectedNonSolvedIndices.remove(foundIndex)

        # Folder valid, check it
        globalLogger.log(logging.INFO, f"Checking index {foundIndex}, folder {rgbFolder}...")
        for filename in os.listdir(rgbFolder):
            assert filename.endswith(".jpg")
            fullImgPath = os.path.join(rgbFolder, filename)
            try:
                img = Image.open(fullImgPath).convert('RGB')
            except:
                # expected format: img_{frame}_{camIndex}.jpg
                #fileName = fullImgPath[fullImgPath.rfind('/'):]
                globalLogger.log(logging.INFO, f"Found issue {rgbFolder} file {filename} !")
                res = re.findall(r'\d+', filename)
                frame = int(res[0])
                index = int(res[1])
                if segmentName not in outDict:
                    outDict[segmentName] = []
                outDict[segmentName].append(frame)

        numFoldersChecked += 1
        #if numFoldersChecked >= 6:
        #    break
    return outDict, setOfExpectedNonSolvedIndices

if __name__ == "__main__":
    logging.getLogger("Pipeline").setLevel(logging.INFO)
    globalLogger.setLevel(logging.DEBUG)

    # Define some layers for pipelines runs
    ALL_PIPELINES_LIST = [PipelineStages.PIPELINE_RGB,
                          PipelineStages.PIPELINE_SEMANTIC,
                          PipelineStages.PIPELINE_EXTRACTION,
                          PipelineStages.PIPELINE_POINTCLOUD,
                          PipelineStages.PIPELINE_3DRECONSTRUCTION,
                          PipelineStages.PIPELINE_OUTPUT]
    PARALLEL_PIPELINES_LAYER1 = [PipelineStages.PIPELINE_RGB]
    PARALLEL_PIPELINES_LAYER2 = [PipelineStages.PIPELINE_SEMANTIC]
    PARALLEL_PIPELINES_LAYER3 = copy.copy(ALL_PIPELINES_LIST)
    PARALLEL_PIPELINES_LAYER3.remove(PipelineStages.PIPELINE_RGB)
    PARALLEL_PIPELINES_LAYER3.remove(PipelineStages.PIPELINE_SEMANTIC)

    with tf.device('/CPU'):
        aargs = parseArguments()
        PROCESSING_BATCH_SIZE = aargs.batchSize
        PROCESSING_INDEX_START = aargs.indexSTART  # How many batches to process
        PROCESSING_INDEX_END = aargs.indexEND
        layersToRun = aargs.LAYERS_TO_RUN.split(",")

        # Init the output base folder
        pipeline_params.globalParams.reinitParams(baseOutputPath=aargs.fullOutputPath, minimalOutputPath=aargs.cleanOutputPath)
        pipeline_params.globalParams.FORCE_RECOMPUTE = True if aargs.forceRecompute == 1 else False
        pipeline_params.globalParams.USE_GPU_FOR_SEGMENTATION = 0 # Disabled by default
        pipeline_params.globalParams.FRAMEINDEX_MAX = aargs.DEBUG_MAX_FRAME
        pipeline_params.globalParams.FRAMEINDEX_MIN = aargs.DEBUG_MIN_FRAME

        pipeline_params.globalParams.IGNORE_POINTS_IN_CAR_OR_PEDESTRIAN_BBOXES = 0 if aargs.outputFullCloudReconstruction == 1 else 1
        pipeline_params.globalParams.BBOX_EXTENSION_FOR_PEDESTRIANS_AND_CARS = aargs.overridePedAndCarsBboxExtension
        pipeline_params.globalParams.POINT_CLOUD_FOR_MOTION_FRAMES = True if aargs.pointCloudForMotionFrames else False
        pipeline_params.globalParams.KEEP_ORIGINAL_FLOATING_POINTS = True if aargs.keepOriginalFloatingPoints else False

        pipeline_params.globalParams.NOISE_FILTERING_WITH_KNN = True if aargs.noiseFilteringWithKNN else False
        pipeline_params.globalParams.NOISE_FILTERING_WITH_KNNStatistical = True if aargs.noiseFilteringWithKNNStatistical else False
        pipeline_params.globalParams.NOISE_FILTERING_WITH_VOXELIZATION = True if aargs.noiseFilteringWithVoxelization else False

        pipeline_params.globalParams.SCALE_USED_IN_FINAL_RECONSTRUCTION = float(aargs.scaleUsedInFinalReconstruction)

        # Use these debug features to cut only sections of data
        allSortedScenesDesc = pd.read_csv(aargs.scenesFile)
        allSortedScenesDesc = allSortedScenesDesc[allSortedScenesDesc["PedestriansDensity"] >= MIN_DENSITY_TO_USE_SCENE]
        allSortedScenesDesc = list(allSortedScenesDesc["Paths"])
        allSortedScenesDesc = [scenePath.replace('\n','').replace('\t','') for scenePath in allSortedScenesDesc]

        # Special case for fixDataset option
        if aargs.fixDataset == 1:
            # Getting a list of path segments and frame indices which are wrong
            RGB_scenes_and_frames, nonSolvedIndices = verifyRGBExtracted(aargs, pipeline_params.globalParams, allSortedScenesDesc)

            if len(nonSolvedIndices):
                globalLogger.info(f"I've detected unsolved indices {nonSolvedIndices}")
                for unsolvedSegmentIndex in nonSolvedIndices:
                    fixingParams = copy.copy(pipeline_params.globalParams)
                    fixingParams.FRAMEINDEX_MAX = fixingParams.FRAMEINDEX_MIN = unsolvedSegmentIndex
                    fixingParams.FORCE_RECOMPUTE = False
                    globalLogger.info(f"Continuing segment index {unsolvedSegmentIndex} path {allSortedScenesDesc[unsolvedSegmentIndex]}")
                    work_list = [allSortedScenesDesc, [unsolvedSegmentIndex], fixingParams,
                                 [PipelineStages.PIPELINE_RGB], 0]
                    workFunc(work_list)
                globalLogger.info(f"RE-Running the proces...")
                RGB_scenes_and_frames, nonSolvedIndices = verifyRGBExtracted(aargs, pipeline_params.globalParams,
                                                                             allSortedScenesDesc)
                assert len(nonSolvedIndices) == 0, f"Still COLDN't solve segment index {unsolvedSegmentIndex} path {allSortedScenesDesc[unsolvedSegmentIndex]}"

            globalLogger.info(f"I've detected broken fix, fixing them now all: {RGB_scenes_and_frames}")
            for item in RGB_scenes_and_frames.items():
                segmentName = item[0]
                frameIndices = item[1]

                # identify the segment path in allSortedScenesDesc...
                taskIndex = -1
                for index, segPath in enumerate(allSortedScenesDesc):
                    if segmentName in segPath:
                        taskIndex = index
                        break
                assert taskIndex != -1, (f"Couldn't find the missing segment name {segmentName}")

                for frameIndex in frameIndices:
                    fixingParams = copy.copy(pipeline_params.globalParams)
                    fixingParams.FRAMEINDEX_MAX = fixingParams.FRAMEINDEX_MIN = frameIndex
                    fixingParams.FORCE_RECOMPUTE = True

                    globalLogger.info(f"Fixing segment name {segmentName} at frame {frameIndex}")
                    work_list = [allSortedScenesDesc, [taskIndex], fixingParams, [PipelineStages.PIPELINE_RGB], 0]
                    workFunc(work_list)

            #print(RGB_scenes_and_frames)
            exit(0)

        PROCESSING_INDEX_END = min(PROCESSING_INDEX_END, len(allSortedScenesDesc)-1)
        processingHeadIndex = PROCESSING_INDEX_START
        batchIndex = 0

        params = pipeline_params.globalParams

        # Is serial running ?
        if aargs.USE_MULTIPROCESSING == 0:
            # Which pipeline stages are requested ?
            PIPELINES_STAGES_REQUESTED = []
            if "1" in layersToRun:
                PIPELINES_STAGES_REQUESTED.extend(PARALLEL_PIPELINES_LAYER1)
            if "2" in layersToRun:
                PIPELINES_STAGES_REQUESTED.extend(PARALLEL_PIPELINES_LAYER2)
            if "3" in layersToRun:
                PIPELINES_STAGES_REQUESTED.extend(PARALLEL_PIPELINES_LAYER3)

        else:
            # Override some params to find resurces globally on workers
            pipeline_params.globalParams.ADE20K_TO_CARLA_MAPPING_CSV = os.path.join(
                "/home/ciprian/cluster-ciprian/Waymo/_DATA/semanticSegmentation/data", "object150_info_TO_CARLA.csv")
            pipeline_params.globalParams.SEGMENTATION_SETUP_DATA_PATH = "/home/ciprian/cluster-ciprian/Waymo/_DATA/semanticSegmentation"
            assert os.path.exists(pipeline_params.globalParams.ADE20K_TO_CARLA_MAPPING_CSV) and os.path.exists(
                pipeline_params.globalParams.SEGMENTATION_SETUP_DATA_PATH)

            # iF the user is running only segmentation, there is no need to use ray for now...
            if not isSegmentationTheOnlyLayer(layersToRun):
                ray.init(address="auto")


        # Any items to process left ?
        while processingHeadIndex <= PROCESSING_INDEX_END:
            thisBatchIndices = list(range(processingHeadIndex, min(processingHeadIndex + PROCESSING_BATCH_SIZE, PROCESSING_INDEX_END + 1)))
            #sortedScenesPaths = allSortedScenesDesc[processingHeadIndex : min(len(allSortedScenesDesc), (processingHeadIndex+BATCH_SIZE))]
            processingHeadIndex = thisBatchIndices[-1] + 1

            if aargs.USE_MULTIPROCESSING == 0:
                # Create a work queue
                work_list = [allSortedScenesDesc, thisBatchIndices, params, PIPELINES_STAGES_REQUESTED, 0]
                # work_list = [sortedScenesPaths, params, PARALLEL_PIPELINES_LAYER1, 0]
                workFunc(work_list)
            else:
                # Get the assignment work for each cpu and gpu
                numActors_cpu = aargs.NUM_ACTORS_CPU #20
                taskAssignment_cpu = getTaskAssignment(thisBatchIndices, numActors_cpu)
                numActors_gpu = aargs.NUM_ACTORS_GPU
                taskAssignment_gpu = [] if numActors_gpu == 0 else getTaskAssignment(thisBatchIndices, numActors_gpu)

                # Three layers:
                # (1) RGB  (2) Seg  (3) The rest

                # Layer 1 setup
                #######
                # Create a work queue for stage 1
                def layer1():
                    simulators = [Simulator.remote([allSortedScenesDesc, taskAssignment_cpu[i], params, PARALLEL_PIPELINES_LAYER1,i]) for i in range(numActors_cpu)]
                    results = ray.get([s.simulate.remote() for s in simulators])

                def layer2():
                    print("GPU TASK ASSIGNMENT is ", taskAssignment_gpu)

                    # The only way to run segmentation on cluster is to use GPUs. CPU concurrency is blocked by I/O, that's why the commented code below
                    params_gpu = []
                    for indexGpu in range(numActors_gpu):
                        p_gpu = copy.copy(params)
                        p_gpu.USE_GPU_FOR_SEGMENTATION = indexGpu # pipeline_params.globalParams.USE_GPU_FOR_SEGMENTATION # indexGpu # TODO FOR MULTI_GPU 
                        params_gpu.append(p_gpu)

                    work_list = [([allSortedScenesDesc, taskAssignment_gpu[i], params_gpu[i], PARALLEL_PIPELINES_LAYER2, i]) for i in range(numActors_gpu)]

                    #for indexGpu in range(numActors_gpu):
                    #    workFunc(work_list[indexGpu])
                    print("Pool of gpus..")
                    #poolOfGPUs = multiprocessing.Pool(numActors_gpu)
                    #print(poolOfGPUs.map(workFunc, work_list))
                    mp.set_start_method("spawn", force=True)
                    pool = mp.Pool(processes=numActors_gpu)
                    res = pool.map(workFunc, work_list)

                    """
                    params_gpu = []
                    for indexGpu in range(numActors_gpu):
                        p_gpu = copy.copy(params)
                        p_gpu.USE_GPU_FOR_SEGMENTATION = pipeline_params.globalParams.USE_GPU_FOR_SEGMENTATION # indexGpu # TODO FOR MULTI_GPU 
                        params_gpu.append(p_gpu)

                    #print("REALLY OUTPUT", numActors_gpu, pipeline_params.globalParams.USE_GPU_FOR_SEGMENTATION)
                    if pipeline_params.globalParams.USE_GPU_FOR_SEGMENTATION != -1 and numActors_gpu > 0: # Should we use gpu ?
                        #print("Using GPU for segmentation index ", params_gpu[0].USE_GPU_FOR_SEGMENTATION)                    
                        RemoteGPUSimulator = ray.remote(num_gpus=1)(Simulator)
                        simulators = [RemoteGPUSimulator.remote([sortedScenesPaths[taskAssignment_gpu[i][0]:taskAssignment_gpu[i][1]], params_gpu[i], PARALLEL_PIPELINES_LAYER2, i]) for i in range(numActors_gpu)]
                        results = ray.get([s.simulate.remote() for s in simulators])
                    else:
                        FORCE_SINGLEPROCESS=False # Hack for segmentation..
                        p_cpu = copy.copy(params)
                        p_cpu.USE_GPU_FOR_SEGMENTATION = -1
                        if FORCE_SINGLEPROCESS:
                            work_list = [sortedScenesPaths, p_cpu, PARALLEL_PIPELINES_LAYER2, 0]
                            workFunc(work_list)
                        else:
                            simulators = [Simulator.remote([sortedScenesPaths[taskAssignment_cpu[i][0]:taskAssignment_cpu[i][1]], p_cpu, PARALLEL_PIPELINES_LAYER2, i]) \
                                                            for i in range(numActors_cpu)]
                            results = ray.get([s.simulate.remote() for s in simulators])
                    """

                def layer3():
                    simulators = [Simulator.remote([allSortedScenesDesc, taskAssignment_cpu[i], params, PARALLEL_PIPELINES_LAYER3, i]) for i in range(numActors_cpu)]
                    results = ray.get([s.simulate.remote() for s in simulators])

                if ("1" in layersToRun):
                    print("********Starting Layer 1...")
                    layer1()

                
                if ("2" in layersToRun):
                    print("********Starting Layer 2...")
                    layer2()

                if ("3" in layersToRun):
                    print("********Starting Layer 3...")
                    layer3()

                # Create a resource pool for multi processing
                #resourcesPool = multiprocessing.Pool(1) # multiprocessing.cpu_count())
                #resourcesPool.map(workFunc, work_list)
                #resourcesPool.join()

                #print(results)

