# The purpose of this pileine stage script is to copy only the cleaned output files that is needed in the end (such that we can share them easily)

import os
import pipeline_commons
import shutil
import ReconstructionUtils

def do_output(segmentPath, globalParams):
    segmentName                 = pipeline_commons.extractSegmentNameFromPath(segmentPath)
    segmentOutputMinimalPath    = os.path.join(globalParams.MINIMAL_OUTPUT_PATH, segmentName)
    segmentOutputFullPath       = os.path.join(globalParams.BASE_OUTPUT_PATH, segmentName)

    if not os.path.exists(segmentOutputMinimalPath):
        os.makedirs(segmentOutputMinimalPath, exist_ok=True)

    # PAIRs of: (filename to copy, optional or not)
    filesToCopyToOutputMin = [(ReconstructionUtils.FILENAME_CARS_TRAJECTORIES, False),
                              (ReconstructionUtils.FILENAME_PEOPLE_TRAJECTORIES, False),
                              (ReconstructionUtils.FILENAME_CARLA_BBOXES, True),
                              (ReconstructionUtils.FILENAME_COMBINED_CARLA_ENV_POINTCLOUD, False),
                              (ReconstructionUtils.FILENAME_COMBINED_CARLA_ENV_POINTCLOUD_SEGCOLOR, True),
                              (ReconstructionUtils.FILENAME_CENTERING_ENV, False),
                              (ReconstructionUtils.FILENAME_CAMERA_INTRISICS, True)
                              ]

    for fileToCopy in filesToCopyToOutputMin:
        optional = fileToCopy[1]
        filename = fileToCopy[0]

        srcFullFilePath = os.path.join(segmentOutputFullPath, filename)
        dstFullFilePath = os.path.join(segmentOutputMinimalPath, filename)

        if os.path.exists(srcFullFilePath) == False:
            if optional == False:
                assert False, (f"Can't copy filename {filename} because it doesn't exists !")
        else:
            shutil.copyfile(srcFullFilePath, dstFullFilePath)

if __name__ == "__main__":
    import pipeline_params
    do_output(pipeline_params.FILENAME_SAMPLE[0], pipeline_params.globalParams)
