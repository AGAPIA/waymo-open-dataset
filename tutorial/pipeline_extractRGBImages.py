# This script is a helper to extract all the RGB images from a WAYMO dataset
# and prepare them as input to segmentation

import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys

tf.enable_eager_execution()
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

import pipeline_commons

def getImagePath(destFolder, frameIndex, cameraIndex):
    return os.path.join(destFolder, f'frame_{frameIndex}_img_{cameraIndex}.jpg')

# Gather all images from a frame to a dictionary
def gatherImagesFromFrame(frameData, frameIndex, datasetPictures, outputFolder, forceRecompute = False):
    localImagesDict = {}

    images = pipeline_commons.getSortedImagesFromFrameData(frameData)
    for index,img in enumerate(images):
        if os.path.exists(getImagePath(outputFolder, frameIndex, index)) and forceRecompute != True:
            continue

        imageBytes = img.image
        imgDecoded = tf.image.decode_jpeg(imageBytes)
        imgDecodedNumpy = imgDecoded.numpy()
        localImagesDict[index] = imgDecodedNumpy

    datasetPictures[frameIndex] = localImagesDict

import shutil

# Save all RGB images in the dictionary { frame index I : { img index i : data } } to files like: frame_I_i.jpg
def saveImagesAsSegmentationInput(allRGBImagesDict, destFolder):
    if not os.path.exists(destFolder):
        os.makedirs(destFolder, mode=0o777, exist_ok=True)

    from PIL import Image
    for frameIndex, imagesPerFrame in allRGBImagesDict.items():
        for camera_index, imgData in imagesPerFrame.items():
            image = Image.fromarray(imgData)
            imagePath = getImagePath(destFolder, frameIndex, camera_index)
            image.save(imagePath)

# Given a list of recoded segments from WAYMO, extract and save the images to semanticSegmentation/INPUT folder
def do_RGBExtraction(segmentPath, globalParams):
    segmentName = pipeline_commons.extractSegmentNameFromPath(segmentPath)
    assert os.path.exists(segmentPath), f'The file you specified {segmentPath} doesn\'t exist !'

    # 1. Iterate over frame by frame of a segment
    dataset = tf.data.TFRecordDataset(segmentPath, compression_type='')

    numFrames = 0
    for index, data in enumerate(dataset):
        numFrames += 1
    print(f"Num frames in {segmentName}: ", numFrames)

    allRGBImagesDict = {}
    for index, data in enumerate(dataset):
        if (globalParams.FRAMEINDEX_MIN > index) and (index != 0): # We need first frame reference point !!:
            continue
        if globalParams.FRAMEINDEX_MAX < index:
            break

        if (index % pipeline_commons.FRAMESINFO_IMGS_DEBUG_RATE == 0):
            pipeline_commons.globalLogger.info(f"Parsing frame {index}/{numFrames}")

        # Read the frame in bytes
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        # Gather and decode all RGB images from this frame to the global store
        segInputFolder = os.path.join(globalParams.SEG_INPUT_IMAGES_BASEPATH, segmentName, globalParams.SEG_INPUT_IMAGES_RGBFOLDER)
        gatherImagesFromFrame(frame, index, allRGBImagesDict, segInputFolder, forceRecompute=globalParams.FORCE_RECOMPUTE)

        saveImagesAsSegmentationInput(allRGBImagesDict, segInputFolder)

# Use do_extraction from exterior and let main just for testing purposes
# Or refactor the code with argparse
if __name__ == "__main__":  # 'frames']
    import pipeline_params
    pipeline_params.globalParams.FRAMEINDEX_MIN = 0
    pipeline_params.globalParams.FRAMEINDEX_MAX = 5

    do_RGBExtraction(pipeline_params.FILENAME_SAMPLE[0], pipeline_params.globalParams)

