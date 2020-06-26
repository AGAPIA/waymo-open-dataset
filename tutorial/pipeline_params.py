import os

class GlobalParams():
    # This initializes to default output path (Dev mode )
    def __init__(self, baseOutputPath = os.path.join("semanticSegmentation", "OUTPUT"), minimalOutputPath = os.path.join("semanticSegmentation", "OUTPUTMIN")):
        self.reinitParams(baseOutputPath, minimalOutputPath)

    # You can overwrite it using this func
    def reinitParams(self, baseOutputPath, minimalOutputPath):
        # Where to write the output of scenes filtering tool
        self.OUTPUT_SCENES_FILTERING_PATH = os.path.join(baseOutputPath, r'scenes.csv')

        # WHere to write the output of the data converion process
        self.BASE_OUTPUT_PATH       = baseOutputPath
        self.MINIMAL_OUTPUT_PATH    = minimalOutputPath

        # Mapping from ADE20K, the segmentation method dataset that we use to CARLA - the output
        ##################
        self.ADE20K_TO_CARLA_MAPPING_CSV = os.path.join("semanticSegmentation", "data", "object150_info_TO_CARLA.csv") # Relative to this script folder

        # Input output for RGB extraction / segmentation
        self.SEG_INPUT_IMAGES_BASEPATH = self.BASE_OUTPUT_PATH
        self.SEG_INPUT_IMAGES_RGBFOLDER = "CameraRGB"

        self.SEG_OUTPUT_LABELS_BASEFILEPATH = self.BASE_OUTPUT_PATH
        self.SEG_OUTPUT_LABELS_SEGFOLDER = "CameraSeg"

        self.SEG_OUTPUTCOMP_LABELS_BASEFILEPATH = self.SEG_OUTPUT_LABELS_BASEFILEPATH # os.path.join(SEG_OUTPUT_LABELS_BASEFILEPATH,
        self.SEG_OUTPUTCOMP_LABELS_RGBFOLDER = "RGBCOMP"
        #SEG_OUTPUT_LABELS_FILENAME = "_labels.pkl"

        # Where to save the output for motion data (e.g. cars and people trajectories)
        self.MOTION_OUTPUT_BASEFILEPATH = self.BASE_OUTPUT_PATH

        # Where to save the output for point cloud reconstruction
        self.POINTCLOUD_OUTPUT_BASEFILEPATH = self.BASE_OUTPUT_PATH

        self.FRAMEINDEX_MIN = 0
        self.FRAMEINDEX_MAX = 199 # Some defaults...

        # How many frames modulo to skip for reconstruction (if needed
        self.FRAMES_SKIP = 1

        # Even if the resource exists, should I recompute them ?
        self.FORCE_RECOMPUTE = True

        # If you really want to use GPU for segmentation...if you have a strong one :)
        self.USE_GPU_FOR_SEGMENTATION = True

globalParams = GlobalParams()


# A few sample segment paths to test pipeline
FILENAME_SAMPLE = ["/home/ciprian/Downloads/Waymo/segment-10023947602400723454_1120_000_1140_000_with_camera_labels.tfrecord"]
