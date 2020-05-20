# A set of utils for processing Waymo library,common for all pipeline stages
import os
import re

# Input output for RGB extraction / segmentation
SEG_INPUT_IMAGES_BASEPATH = os.path.join("semanticSegmentation", "TEST_INPUT")
SEG_OUTPUT_LABELS_BASEFILEPATH = os.path.join("semanticSegmentation", "TEST_OUTPUT")
SEG_OUTPUTCOMP_LABELS_BASEFILEPATH = os.path.join("semanticSegmentation", "TEST_OUTPUTCOMP")
SEG_OUTPUT_LABELS_FILENAME = "_labels.pkl"

# Extracts a segment name from a given path
def extractSegmentNameFromPath(S):
    # Expecting the path (S) to be something like "/home/ciprian/Downloads/Waymo/segment-10023947602400723454_1120_000_1140_000_with_camera_labels.tfrecord"
    # then we store only 10023947602400723454_1120_000_1140_000 out of this
    return S[S.rfind('/') + 1 + len("segment-"): S.find("_with")]


# A few sample segment paths to test pipeline
FILENAME_SAMPLE = ["/home/ciprian/Downloads/Waymo/segment-10023947602400723454_1120_000_1140_000_with_camera_labels.tfrecord"]
