"""
 The purpose of this script is to analyze a given folders of sequences and check which one of them have the followings:
 - Filter that dataset to have at least 3 Pedestrians which are most 100m distance from the car
 - Based on this info, it will classify sort the sequences by pedestrians count

 - Input: a folder with Waymo data (will be recusrively analyzed)
 - Output:  The segments name/paths, sorted by how interesting is the scene (how many pedestrians / density are close to the car)
"""

MIN_PEDESTRIANS = 2
MAX_DISTANCE = 50
MAX_DISTANCE_SQR = (MAX_DISTANCE*MAX_DISTANCE)
MIN_DENSITY = 0

import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
from pipeline_commons import *
import pipeline_params
import pickle
from pathlib import Path
import pandas as pd
import pickle

#tf.enable_eager_execution()
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset


# Returns the number of different pedestrians and the average density of pedestrians per frame in the scene
def scoreScene(fullPath):
    # 1. Iterate over frame by frame of a segment
    dataset = tf.data.TFRecordDataset(fullPath.__str__(), compression_type='')

    feasiblePedestrianIds = {} # pedestrian id : how many frames it occures

    numFramesInSegment = 0
    for frameIndex, data in enumerate(dataset):
        numFramesInSegment += 1

        if frameIndex % 100 == 0:
            print("..Processing frame index ", frameIndex)

        # Read the frame in bytes
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        lidarLabels = frame.laser_labels

        for label in lidarLabels:
            if label.type != label.TYPE_PEDESTRIAN:
                continue

            pedId = label.id
            box = label.box
            x, y, z = box.center_x, box.center_y, box.center_z
            # Remember that position of pedestrian is in vehicle space
            if (x*x + y*y + z*z) > MAX_DISTANCE_SQR:
                continue

            if feasiblePedestrianIds.get(pedId) is None:
                feasiblePedestrianIds[pedId] = 0
            feasiblePedestrianIds[pedId] += 1

    totalNumFramesWithPedestrians = 0
    for pedId, numFrames in feasiblePedestrianIds.items():
        totalNumFramesWithPedestrians += numFrames

    densityPerFrame = totalNumFramesWithPedestrians / numFramesInSegment

    return len(feasiblePedestrianIds), densityPerFrame

def doFindInterestingScenes(baseDatasetFolders):
    mapSceneToScore = {} # 'scenepath' : score
    total_processed = 0
    for folder in baseDatasetFolders:
        print(f"Analyzing folder {folder}")
        print("===================================")
        for path in Path(folder).rglob('*.tfrecord'):
            total_processed += 1
            #if total_processed > 4:
            #    break

            #print(path.name)
            fullPath = path.absolute()
            #print(fullPath)
            print(f"#### Analyzing segment path index {total_processed} from path {fullPath}")

            numPed, pedDensity = scoreScene(fullPath)
            if numPed >= MIN_PEDESTRIANS and pedDensity > MIN_DENSITY: # Is it valid ?
                mapSceneToScore[fullPath] = (numPed, pedDensity)

            print(f"Score {fullPath} is: numPedestrians = {numPed} pedestrians density = {pedDensity}")

    print(f"Processed {total_processed} scenes. Starting to save the final csv...")

    with open("temp_mapSceneToScore", "wb") as tempFile:
        pickle.dump(mapSceneToScore, tempFile, protocol=2)

    mapSceneToScore = sorted(mapSceneToScore.items(), key=lambda item: item[1][1], reverse=True) # Sorting by density

    scenesScoringData = { 'Paths' : [v[0] for v in mapSceneToScore],
                          'NumPedestrians' : [v[1][0] for v in mapSceneToScore],
                          'PedestriansDensity' : [v[1][1] for v in mapSceneToScore]}
    scenesScoringDataFrame = pd.DataFrame(scenesScoringData, columns=['Paths', 'NumPedestrians', 'PedestriansDensity'])
    scenesScoringDataFrame.to_csv (pipeline_params.globalParams.OUTPUT_SCENES_FILTERING_PATH, index = False, header=True)

if __name__ == "__main__":
    pipeline_params.globalParams.OUTPUT_SCENES_FILTERING_PATH = "scenes.csv"
    doFindInterestingScenes([sys.argv[1]])
