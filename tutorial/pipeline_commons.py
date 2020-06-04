# A set of utils for processing Waymo library,common for all pipeline stages
import os
import re

BASE_OUTPUT_PATH = os.path.join("semanticSegmentation", "OUTPUT")

# Input output for RGB extraction / segmentation
SEG_INPUT_IMAGES_BASEPATH = BASE_OUTPUT_PATH
SEG_INPUT_IMAGES_RGBFOLDER = "CameraRGB"

SEG_OUTPUT_LABELS_BASEFILEPATH = BASE_OUTPUT_PATH
SEG_OUTPUT_LABELS_SEGFOLDER = "CameraSeg"

SEG_OUTPUTCOMP_LABELS_BASEFILEPATH = SEG_OUTPUT_LABELS_BASEFILEPATH # os.path.join(SEG_OUTPUT_LABELS_BASEFILEPATH,
SEG_OUTPUTCOMP_LABELS_RGBFOLDER = "RGBCOMP"
#SEG_OUTPUT_LABELS_FILENAME = "_labels.pkl"

# Where to save the output for motion data (e.g. cars and people trajectories)
MOTION_OUTPUT_BASEFILEPATH = BASE_OUTPUT_PATH

# Where to save the output for point cloud reconstruction
POINTCLOUD_OUTPUT_BASEFILEPATH = BASE_OUTPUT_PATH

# Extracts a segment name from a given path
def extractSegmentNameFromPath(S):
    # Expecting the path (S) to be something like "/home/ciprian/Downloads/Waymo/segment-10023947602400723454_1120_000_1140_000_with_camera_labels.tfrecord"
    # then we store only 10023947602400723454_1120_000_1140_000 out of this
    return S[S.rfind('/') + 1 + len("segment-"): S.find("_with")]


# A few sample segment paths to test pipeline
FILENAME_SAMPLE = ["/home/ciprian/Downloads/Waymo/segment-10023947602400723454_1120_000_1140_000_with_camera_labels.tfrecord"]
