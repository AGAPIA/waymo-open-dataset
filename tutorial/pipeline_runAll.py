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

import logging
from pipeline_commons import globalLogger

def parseArguments():
    parser = argparse.ArgumentParser(description="Pipeline process for extracting Waymo data")
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

    aargs = parser.parse_args(sys.argv[1:])
    return aargs

if __name__ == "__main__":
    logging.getLogger("Pipeline").setLevel(logging.INFO)
    globalLogger.setLevel(logging.DEBUG)

    with tf.device('/CPU:0'):
        aargs = parseArguments()

        # Init the output base folder
        pipeline_params.globalParams.reinitParams(baseOutputPath=aargs.fullOutputPath, minimalOutputPath=aargs.cleanOutputPath)
        pipeline_params.globalParams.FORCE_RECOMPUTE = True if aargs.forceRecompute == 1 else False
        pipeline_params.globalParams.USE_GPU_FOR_SEGMENTATION = False # Disabled because mine is weak (too low on memory..)

         # Use these debug features to cut only sections of data

        # Read the segment to analyze on this run
        with open(aargs.scenesFile) as scenesHandleFile:
            segmentPath = scenesHandleFile.readline()
            cnt = 0
            while segmentPath:
                segmentPath = segmentPath.replace('\n','').replace('\t','')
                if segmentPath == '' or segmentPath== ' ':
                    break

                #print("Line {}: {}".format(cnt, segmentPath.strip()))
                globalLogger.info((f"$$$$$$$$$$$$ Processing segment index: {cnt}, path: {segmentPath} $$$$$$$$$$$$"))
                start_time = time.time()
                numFrames = pipeline_commons.getNumFramesInSegmentPath(segmentPath)
                pipeline_params.globalParams.FRAMEINDEX_MAX = min(aargs.DEBUG_MAX_FRAME, numFrames-1)
                pipeline_params.globalParams.FRAMEINDEX_MIN = max(0, aargs.DEBUG_MIN_FRAME)
                globalLogger.info((f"There are {numFrames} frames. Starting the pipeline process between frames {pipeline_params.globalParams.FRAMEINDEX_MIN}-{pipeline_params.globalParams.FRAMEINDEX_MAX}, recompute: {pipeline_params.globalParams.FORCE_RECOMPUTE}"))

                globalLogger.log(logging.INFO,("##Stage 1: RGB Image extraction"))
                pipeline_extractRGBImages.do_RGBExtraction(segmentPath, pipeline_params.globalParams)

                globalLogger.log(logging.INFO,"##Stage 2: Segmentation")
                pipeline_segmentation.runSegmentationOps(segmentPath, pipeline_params.globalParams)

                #time.sleep(10)

                globalLogger.log(logging.INFO,("##Stage 3: Peoples and vehicle trajectories extraction"))
                pipeline_extractPeopleAndCars.do_PeopleAndVehiclesExtraction(segmentPath, pipeline_params.globalParams)

                globalLogger.log(logging.INFO,("##Stage 4: Point cloud reconstruction"))
                pipeline_pointCloudReconstruction.do_PointCloudReconstruction(segmentPath, pipeline_params.globalParams)

                globalLogger.log(logging.INFO,("##Stage 5: 3D Scene reconstruction"))
                pipeline_3DReconstruction.do_3DReconstruction(segmentPath, pipeline_params.globalParams)

                globalLogger.log(logging.INFO,("##Stage 6: Copy files to output"))
                pipeline_outputProcessing.do_output(segmentPath, pipeline_params.globalParams)

                end_time = time.time()
                globalLogger.log(logging.INFO,("$$$ Processing finished, time spent", (end_time-start_time)))

                segmentPath = scenesHandleFile.readline()
                cnt += 1

