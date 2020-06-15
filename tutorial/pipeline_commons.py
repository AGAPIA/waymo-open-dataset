# A set of utils for processing Waymo library,common for all pipeline stages
import os
import re
import sys
sys.path.append("/home/ciprian/Work/RLAgent/commonUtils") #os.path.join(os.path.dirname(__file__), "lib"))

# Where to write the output of scenes filetering tool
OUTPUT_SCENES_FILTERING_PATH = r'scenes.csv'

# WHere to write the output of the data converion process
BASE_OUTPUT_PATH = os.path.join("semanticSegmentation", "OUTPUT")

# Mapping from ADE20K, the segmentation method dataset that we use to CARLA - the output
##################
ADE20K_TO_CARLA_MAPPING_CSV = os.path.join("semanticSegmentation", "data", "object150_info_TO_CARLA.csv")

# 0 	None		Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , ),
# 1 	Buildings	Label(  ''             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
# 2 	Fences		Label(  ''                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
# 3 	Other		Label(  ''               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
# 4 	Pedestrians	Label(  ''               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
# 5 	Poles		Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
# 6 	RoadLines	Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
# 7 	Roads		Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
# 8 	Sidewalks	Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
# 9 	Vegetation	Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
# 10 	Vehicles	Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
# 11 	Walls		Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
# 12 	TrafficSigns	Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),


####################

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

# Unified way to process the camera in a frame in a ordered way
def getSortedImagesFromFrameData(frame):
    images = sorted(frame.images, key=lambda i:i.name)
    return images

# Extracts a segment name from a given path
def extractSegmentNameFromPath(S):
    # Expecting the path (S) to be something like "/home/ciprian/Downloads/Waymo/segment-10023947602400723454_1120_000_1140_000_with_camera_labels.tfrecord"
    # then we store only 10023947602400723454_1120_000_1140_000 out of this
    return S[S.rfind('/') + 1 + len("segment-"): S.find("_with")]


# A few sample segment paths to test pipeline
FILENAME_SAMPLE = ["/home/ciprian/Downloads/Waymo/segment-10023947602400723454_1120_000_1140_000_with_camera_labels.tfrecord"]
FOLDER_WAYMODATASET_SAMPLE = ["/home/ciprian/Downloads/Waymo/"] # Where the entire dataset is for Wamo