"""
The purpose of this stage is to use:
 1. Use the segmentation files and data to create a single unified carla_combined_moving describing the full environment
 2. COpy only the needed files to another folder - with the intention to share less data in general.

"""

import os, sys
import numpy as np
sys.path.append("/home/ciprian/Work/RLAgent/commonUtils") #os.path.join(os.path.dirname(__file__), "lib"))

import ReconstructionUtils
import pipeline_commons

# Do reconstruction between FRAME_MIN and FRAME_MAX
# frameSkip = How many frames modulo to skip for reconstruction (if needed
def do_3DReconstruction(segmentPath, globalParams):
    segmentName = pipeline_commons.extractSegmentNameFromPath(segmentPath)
    folderOutput = os.path.join(globalParams.POINTCLOUD_OUTPUT_BASEFILEPATH, segmentName)

    class DatasetOptions:
        def __init__(self):
            self.dataAlreadyInWorldSpace = True
            self.framesIndicesCaptured = list(np.arange(int(globalParams.FRAMEINDEX_MIN), int(globalParams.FRAMEINDEX_MAX) + 1))  # TODO: fix number of frames correctly by reading metadata !
            self.LIMIT_FRAME_NUMBER = globalParams.FRAMEINDEX_MAX
            # How many frames modulo to skip for reconstruction (if needed
            self.frameSkip = globalParams.FRAMES_SKIP
            self.numInitialFramesToIgnore = (globalParams.FRAMEINDEX_MIN - 1)  # How many frames in the beginning of the scene to ignore

    options = DatasetOptions()
    reconstruction, people_rec, cars_rec, scale, ped_dict, cars_2D, people_2D, valid_ids, cars_dict, init_frames, init_frames_cars = ReconstructionUtils.reconstruct3D_ply(
        folderOutput, recalculate=globalParams.FORCE_RECOMPUTE, datasetOptions=options)


if __name__ == "__main__":
    import pipeline_params
    pipeline_params.globalParams.FRAMEINDEX_MIN = 0
    pipeline_params.globalParams.FRAMEINDEX_MAX = 5
    segmentName     = pipeline_commons.extractSegmentNameFromPath(pipeline_params.FILENAME_SAMPLE[0])

    do_3DReconstruction(segmentName, pipeline_params.globalParams)

