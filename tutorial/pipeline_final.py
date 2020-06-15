"""
The purpose of this stage is to use:
 1. Use the segmentation files and data to create a single unified carla_combined_moving describing the full environment
 2. COpy only the needed files to another folder - with the intention to share less data in general.

"""

import os, sys
import numpy as np
sys.path.append("/home/ciprian/Work/RLAgent/commonUtils") #os.path.join(os.path.dirname(__file__), "lib"))

from ReconstructionUtils import  reconstruct3D_ply
from pipeline_commons import *


if __name__ == "__main__":
    segmentName     = extractSegmentNameFromPath(FILENAME_SAMPLE[0])
    folderOutput    = os.path.join(POINTCLOUD_OUTPUT_BASEFILEPATH, segmentName)

    LIMIT_FRAME_NUMBER = 200

    class DatasetOptions:
        def __init__(self):
            self.dataAlreadyInWorldSpace = True
            self.framesIndicesCaptured = list(np.arange(0, min(199,
                                                               LIMIT_FRAME_NUMBER)))  # TODO: fix number of frames correctly by reading metadata !
            self.LIMIT_FRAME_NUMBER = LIMIT_FRAME_NUMBER
            # How many frames modulo to skip for reconstruction (if needed
            self.frameSkip = 1
            self.numInitialFramesToIgnore = 0 # How many frames in the beginning of the scene to ignore


    options = DatasetOptions()
    reconstruction, people_rec, cars_rec, scale, ped_dict, cars_2D, people_2D, valid_ids, cars_dict, init_frames, init_frames_cars = reconstruct3D_ply(folderOutput, recalculate=True, datasetOptions=options)

