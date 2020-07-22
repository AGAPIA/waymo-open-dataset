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

        # Some metadata needed for segmentation process
        self.SEGMENTATION_SETUP_DATA_PATH = "semanticSegmentation" # default folder...

        # If you really want to use GPU for segmentation...if you have a strong one :)
        self.USE_GPU_FOR_SEGMENTATION = 0 # Id 0 by default

        # Whether to ignore or not cars and pedestrians in point cloud reconstructions
        self.IGNORE_POINTS_IN_CAR_OR_PEDESTRIAN_BBOXES = True

        self.BBOX_EXTENSION_FOR_PEDESTRIANS_AND_CARS = 1.25

        # see args for doc
        self.POINT_CLOUD_FOR_MOTION_FRAMES = False

        # Keep original floating points.
        # Otherwise we do voxelization which will lower the quality but increase speed
        # This should be used mostly for rendering / vis purposes
        self.KEEP_ORIGINAL_FLOATING_POINTS = False

        # Filter the noise either with VOXElization or KNN. voxelization only rmeove noisy points, Knn also votes for labels
        self.NOISE_FILTERING_WITH_VOXELIZATION = False
        self.NOISE_FILTERING_WITH_KNN = False
        self.NOISE_FILTERING_WITH_KNNStatistical = False

        # The scape up/down value (base is 1.0) used to recompose the final ply scenes from frames ply files
        self.SCALE_USED_IN_FINAL_RECONSTRUCTION = 1.0

globalParams = GlobalParams()


# A few sample segment paths to test pipeline
FILENAME_SAMPLE = ["/home/ciprian/Downloads/Waymo/segment-10023947602400723454_1120_000_1140_000_with_camera_labels.tfrecord"]
