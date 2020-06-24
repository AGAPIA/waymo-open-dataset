# A set of utils for processing Waymo library,common for all pipeline stages
import os
import re
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
#tf.get_logger().setLevel('ERROR')

import logging
#logging.basicConfig(level=logging.INFO)

sys.path.append("../commonUtils") #os.path.join(os.path.dirname(__file__), "lib"))

# Note: the semantic segmentation code has its own logger defined in utils.py/setup_logger func
globalLogger = None
FRAMESINFO_IMGS_DEBUG_RATE = 50 # At which rate to show log info about rgb images processing
def setupLogging():
    global globalLogger
    globalLogger = logging.getLogger("Pipeline")
    globalLogger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('logfile.txt')
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    globalLogger.addHandler(file_handler)
    globalLogger.addHandler(stdout_handler)

setupLogging()

# Unified way to process the camera in a frame in a ordered way
def getSortedImagesFromFrameData(frame):
    images = sorted(frame.images, key=lambda i:i.name)
    return images

# Extracts a segment name from a given path
def extractSegmentNameFromPath(S):
    # Expecting the path (S) to be something like "/home/ciprian/Downloads/Waymo/segment-10023947602400723454_1120_000_1140_000_with_camera_labels.tfrecord"
    # then we store only 10023947602400723454_1120_000_1140_000 out of this
    return S[S.rfind('/') + 1 + len("segment-"): S.find("_with")]

def getNumFramesInSegmentPath(segmentPath):
    segmentName = extractSegmentNameFromPath(segmentPath)
    assert os.path.exists(segmentPath), f'The file you specified {segmentPath} doesn\'t exist !'

    # 1. Iterate over frame by frame of a segment
    dataset = tf.data.TFRecordDataset(segmentPath, compression_type='')

    numFrames = 0
    for index, data in enumerate(dataset):
        numFrames += 1
    #print(f"Num frames in {segmentName}: ", numFrames)
    return numFrames

