# THIS IS A COPY FROM RLAgent
# DO NOT MODIFY IT DIRECTLY OR KEEP IN SYNC ALWAYS WITH THE ONE IN RL AGENT !!!!!!!!!!!

from plyfile import PlyData, PlyElement
import glob
import os.path
import numpy as np
import pickle
import csv
import scipy as sc
import math

# Coordinate change between vehicle and camera.
# Note: this is for cityscapes, with forward axis on X, Z up.
# Basically camera space from Carla to Citiscapes (see their calibration document).
# They use a different camera system. Sequence of operations:
#   cityscapes_camera_pos = P_inv * carla_camera_pos * [-1, -1, 1]
# BUT don't forget that carla data in our dataset has all points in camera space (which means that they are like    pos in camera space = (pos in vehicle space ) * P
# To transform them back to vehicle space =>  pos in camera space * P_inv
P = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])  # vehicle coord to camera
P_inv = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])  # camera to vehicle

# These are the labels mapping from Carla to Citiscapes
# NOTE: we are using by default Carla segmentation label because it is the most simplest mapping possible
# So every dataset we get should map and save their data in Carla space !!!
# 0 	None		Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
# 1 	Buildings	Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
# 2 	Fences		Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
# 3 	Other		Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
# 4 	Pedestrians	Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
# 5 	Poles		Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
# 6 	RoadLines	Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
# 7 	Roads		Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
# 8 	Sidewalks	Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
# 9 	Vegetation	Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
# 10 	Vehicles	Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
# 11 	Walls		Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
# 12 	TrafficSigns	Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),

carla_label_colours=[(0, 0, 0),        #0
                    ( 70, 70, 70) ,     #1
                    (190,153,153) ,     #2
                    (  0,  0,  0),      #3
                    (220, 20, 60),      #4
                    (153,153,153),      #5
                    (128, 64,128),      #6
                    (128, 64,128),      #7
                    (244, 35,232),      #8,
                    (107,142, 35),      #9,
                    (  0, 0, 142),      #10
                    (102,102,156),      #11
                    (220,220,  0),      #12
                     ]

carla_labels=['unlabeled',      #0
                'building' ,    #1
                'fence' ,       #2
                'static',       #3
                'person',       #4
                'pole',         #5
                'road',         #6
                'road',         #7
                'sidewalk',     #8
                'vegetation',   #9
                'car',          #10
                'wall',         #11
                'traffic sign', #12
              ]

# Mapping from CARLA labels to CITYSCAPES labels
def get_label_mapping():
    label_mapping = {}
    label_mapping[0] = 0  # None
    label_mapping[1] = 11  # Building
    label_mapping[2] = 13  # Fence
    label_mapping[3] = 4  # Other/Static
    label_mapping[4] = 24  # Pedestrian
    label_mapping[5] = 17  # Pole
    label_mapping[6] = 7  # RoadLines
    label_mapping[7] = 7  # Road
    label_mapping[8] = 8  # Sidewalk
    label_mapping[9] = 21  # Vegetation
    label_mapping[10] = 26  # Vehicles
    label_mapping[11] = 12  # Wall
    label_mapping[12] = 20  # Traffic sign
    return label_mapping


cityscapes_colours = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
                      (111, 74, 0), (81, 0, 81), (128, 64, 128), (244, 35, 232), (250, 170, 160),
                      (230, 150, 140), (70, 70, 70), (102, 102, 156), (190, 153, 153), (180, 165, 180),
                      (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153), (250, 170, 30),
                      (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
                      (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 0, 90),
                      (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142)]


cityscapes_labels=['unlabeled', 'ego vehicle' , 'rectification border' , 'out of roi' , 'static', 'dynamic', 'ground' , 'road' , 'sidewalk' ,
        'parking' ,'rail track','building' ,'wall' ,'fence', 'guard rail', 'bridge', 'tunnel' ,'pole', 'polegroup',
        'traffic light','traffic sign' ,'vegetation', 'terrain' ,'sky','person' ,'rider', 'car','truck' ,'bus',
        'caravan' ,'trailer', 'train' , 'motorcycle' ,'bicycle','license plate']




NUM_LABELS_CARLA = 13

NO_LABEL_POINT = 0 # ANy database probably...
OTHER_SEGMENTATION_POINT_CARLA = 3
SEGLABEL_PEDESTRIAN=4
SEGLABEL_CAR=10

FILENAME_CARS_TRAJECTORIES = 'cars.p'
FILENAME_PEOPLE_TRAJECTORIES = 'people.p'
FILENAME_CARLA_BBOXES = 'carla_bboxes.csv'
FILENAME_COMBINED_CARLA_ENV_POINTCLOUD = 'combined_carla_moving.ply'
FILENAME_COMBINED_CARLA_ENV_POINTCLOUD_SEGCOLOR = 'combined_carla_moving_segColor.ply'
FILENAME_CENTERING_ENV = 'centering.p'
FILENAME_CAMERA_INTRISICS = 'camera_intrinsics.p'

def isUsefulLabelForReconstruction(label):
    return (label != NO_LABEL_POINT and label != OTHER_SEGMENTATION_POINT_CARLA) and (not isLabelIgnoredInReconstruction(label)) #and (label not in [1])

def isCloudPointRoadOrSidewalk(pos, label):
    return (label > 5 and label < 9)

def isLabelIgnoredInReconstruction(labels):
    return (SEGLABEL_PEDESTRIAN in labels or SEGLABEL_CAR in labels) if isinstance(labels, list) else (labels == SEGLABEL_PEDESTRIAN or labels == SEGLABEL_CAR)

# NOTE: scale = 5 represents conversion from camera coordinate system to because before the coordinate system was in meters, 1 * 5

# Given a dictionary of [3d points from the cloud] -> {rgb segmentation label , raw segmentation label}, save it to the specified file
# (Taken from CARLA's code)
def save_3d_pointcloud(points_3d, filename):
    """Save this point-cloud to disk as PLY format."""

    def construct_ply_header():
        """Generates a PLY header given a total number of 3D points and
        coloring property if specified
        """
        points = len(points_3d)  # Total point number
        header = ['ply',
                  'format ascii 1.0',
                  'element vertex {}',
                  'property float32 x',
                  'property float32 y',
                  'property float32 z',
                  'property uchar diffuse_red',
                  'property uchar diffuse_green',
                  'property uchar diffuse_blue',
                  'property uchar label',
                  'end_header']
        return '\n'.join(header).format(points)

    for point in points_3d:
        label = point[6]
        for p in point:
            try:
                n = float(p)
            except ValueError:
                print ("Problem " + str(point))
    # points_3d = np.concatenate(
    #     (point_list._array, self._color_array), axis=1)
    try:
        ply = '\n'.join(
            ['{:.2f} {:.2f} {:.2f} {:.0f} {:.0f} {:.0f} {:.0f}'.format(*p) for p in points_3d])  # .replace('.',',')
    except ValueError:
        for point in points_3d:
            print (point)
            print ('{:.2f} {:.2f} {:.2f} {:.0f} {:.0f} {:.0f} {:.0f}'.format(*point))
    # Create folder to save if does not exist.
    # folder = os.path.dirname(filename)
    # if not os.path.isdir(folder):
    #     os.makedirs(folder)

    # Open the file and save with the specific PLY format.
    with open(filename, 'w+') as ply_file:
        ply_file.write('\n'.join([construct_ply_header(), ply]))


def save_3d_pointcloud_asSegLabel(points_3d, filename):
    points_3d_seg = [None] * len(points_3d)
    """Save this point-cloud to disk as PLY format."""

    def construct_ply_header():
        """Generates a PLY header given a total number of 3D points and
        coloring property if specified
        """
        points = len(points_3d)  # Total point number
        header = ['ply',
                  'format ascii 1.0',
                  'element vertex {}',
                  'property float32 x',
                  'property float32 y',
                  'property float32 z',
                  'property uchar diffuse_red',
                  'property uchar diffuse_green',
                  'property uchar diffuse_blue',
                  'property uchar label',
                  'end_header']
        return '\n'.join(header).format(points)

    for index, origPoint in enumerate(points_3d):
        # Override the RGB to the RGB of the segmentation
        label = origPoint[6]
        try:
            colorForLabel = cityscapes_colours[label]
        except:
            assert False, "invalid label or something"
            colorForLabel = (0, 0, 0)
        point_seg = (origPoint[0], origPoint[1], origPoint[2], colorForLabel[0], colorForLabel[1], colorForLabel[2], label)
        points_3d_seg[index] = point_seg

        for p in origPoint:
            try:
                n = float(p)
            except ValueError:
                print ("Problem " + str(origPoint))
    # points_3d = np.concatenate(
    #     (point_list._array, self._color_array), axis=1)
    try:
        ply = '\n'.join(
            ['{:.2f} {:.2f} {:.2f} {:.0f} {:.0f} {:.0f} {:.0f}'.format(*p) for p in points_3d_seg])  # .replace('.',',')
    except ValueError:
        for point in points_3d_seg:
            print (point)
            print ('{:.2f} {:.2f} {:.2f} {:.0f} {:.0f} {:.0f} {:.0f}'.format(*point))
    # Create folder to save if does not exist.
    # folder = os.path.dirname(filename)
    # if not os.path.isdir(folder):
    #     os.makedirs(folder)

    # Open the file and save with the specific PLY format.
    with open(filename, 'w+') as ply_file:
        ply_file.write('\n'.join([construct_ply_header(), ply]))

# Read pointcloud from ply file - vertices coord and semnatic labels
#
def read_3D_pointcloud(filename, file_end='/dense/fused_text.ply'):
    plydata_3Dmodel = PlyData.read(filename + file_end)
    nr_points = plydata_3Dmodel.elements[0].count
    pointcloud_3D = np.array([plydata_3Dmodel['vertex'][k] for k in range(nr_points)])
    return pointcloud_3D

# Read a ply file with the correspondences between 3d points and RGB values
def get_rgb_point_cloud(filepath, parts):
    rgb_cloud = os.path.join(filepath, parts[0] + '.ply')
    pointcloud_3D_rgb = read_3D_pointcloud(rgb_cloud, '')
    rgb_dict = {}
    for point in pointcloud_3D_rgb:
        key = (point[0], point[1], point[2])
        rgb_dict[key] = (point[3], point[4], point[5])
    return rgb_dict

#######################

# Given a dictionary of people location in RGB images, save on each row coordinates corresponding to each frame
def save_2D_poses(filepath, people_2D, frames):
    frame_i = 0
    csv_rows = []
    image_sz = [800, 600]
    for frame in sorted(frames):
        frame_name = "%06d.png" % frame
        csv_rows.append([])
        csv_rows[frame_i].append(os.path.join(filepath, 'CameraRGB', frame_name))

        rectangles = []

        for pers_long in people_2D[frame]:
            # print pers_limited
            if len(pers_long) > 0:

                pers = [[min(pers_long[1]), max(pers_long[1])], [min(pers_long[0]), max(pers_long[0])]]
                pers_limited = [[max(int(pers[0][0]), 0), min(int(pers[0][1]), image_sz[1])],
                                [max(int(pers[1][0]), 0), min(int(pers[1][1]), image_sz[0])]]
                if pers_limited[0][1] - pers_limited[0][0] > 10 and pers_limited[1][1] - pers_limited[1][0] > 10:
                    csv_rows[frame_i].append(pers_limited)

        frame_i = frame_i + 1

    with open(os.path.join(filepath, FILENAME_CARLA_BBOXES), 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

# Returns the transformation from world-to-camera space (R_inv/'inverse_rotation'):
# First is rotation, second is translation, third is a list of frame indices where we gathered data
def get_camera_matrix(filepath):
    cameras_path = os.path.join(filepath,
                                'cameras.p')  # Transforms from camera coordinate system to world coordinate system
    cameras_dict = pickle.load(open(cameras_path))
    frames = sorted(cameras_dict.keys())

    #if len(frames) > LIMIT_FRAME_NUMBER:
    #    frames = frames[:LIMIT_FRAME_NUMBER]

    frame = frames[-1]  # np.min(cameras_dict.keys())
    R_inv = cameras_dict[frame]['inverse_rotation']
    middle = R_inv[0:3, 3]
    R = np.transpose(R_inv[0:3, 0:3])
    middle = -np.matmul(R, middle)  # Camera center

    t = R_inv[0:3, 3]

    return R_inv[0:3, 0:3], t, frames  # , car_pos

def get_camera_intrinsic_matrix(filepath):
    cameras_path = os.path.join(filepath, FILENAME_CAMERA_INTRISICS)
    K = pickle.load(open(cameras_path))

    return K

def get_cars_list(list, cars_dict, frames, WorldToCameraRotation, WorldToCameraTranslation,
                  heightOffset, scale, K, filepath, people_flag=False, find_poses=False, datasetOptions=None):
    image_size = [800, 600]
    list_2D = []
    poses = []
    for i, frame in enumerate(sorted(cars_dict.keys())):
        # If there is some dataset options and there is a number of items limit..
        if datasetOptions is not None and i >= datasetOptions.LIMIT_FRAME_NUMBER:
            break

        frame_cars = cars_dict[frame]
        list_2D.append([])
        poses.append([])
        for car_id in frame_cars:

            x_values = []
            y_values = []
            z_values = []
            if False:  # 'bounding_box_coord' in frame_cars[car_id].keys() and not people_flag:
                for point in frame_cars[car_id]['bounding_box_coord']:
                    point = np.squeeze(
                        np.array(np.matmul(WorldToCameraRotation, point.T) + WorldToCameraTranslation).reshape(
                            -1, )) * scale

                    point = point * [-1, -1, 1]  # normal camera coodrinates

                    point = np.squeeze(np.asarray(np.matmul(P_inv, point)))
                    point[2] = point[2] - heightOffset

                    x_values.append(point[0])
                    y_values.append(point[1])
                    z_values.append(point[2])
                car = np.array([[np.min(x_values), np.max(x_values)], [np.min(y_values), np.max(y_values)],
                                [np.min(z_values), np.min(z_values)]])

                x_values = []
                y_values = []
                if find_poses:
                    if 'bounding_box_2D' in frame_cars[car_id].keys():

                        for point in frame_cars[car_id]['bounding_box_2D']:
                            x_values.append(point[0])
                            y_values.append(point[1])

            else:
                dataAlreadyInGlobalSpace = datasetOptions is not None and datasetOptions.dataAlreadyInWorldSpace is True
                if not dataAlreadyInGlobalSpace:
                    car, bboxes_2d = bbox_of_car(K, WorldToCameraRotation, car_id, find_poses, frame_cars, heightOffset, image_size,
                                                 WorldToCameraTranslation, scale)
                else:
                    car, bboxes_2d = bbox_of_entity_alreadyInGlobalSpace(K, WorldToCameraRotation, car_id,
                                                                         find_poses, frame_cars, heightOffset,
                                                                         image_size, WorldToCameraTranslation, scale, people_flag)

                list_2D[i] = bboxes_2d

            if not people_flag:
                car = car.flatten()
                car[1] = car[1] + 1
                car[3] = car[3] + 1
                car[5] = car[5] + 1

            list[i].append(car.astype(int))
    return list, list_2D

# Build a 3D rotation matrix giving yaw
def get_car_rotation_matrix(yaw):
    cy = np.cos(np.radians(yaw))
    sy = np.sin(np.radians(yaw))
    cr = 1
    sr = 0
    cp = 1
    sp = 0
    matrix = np.matrix(np.identity(3))
    matrix[0, 0] = cp * cy
    matrix[0, 1] = cy * sp * sr - sy * cr
    matrix[0, 2] = - (cy * sp * cr + sy * sr)
    matrix[1, 0] = sy * cp
    matrix[1, 1] = sy * sp * sr + cy * cr
    matrix[1, 2] = cy * sr - sy * sp * cr
    matrix[2, 0] = sp
    matrix[2, 1] = -(cp * sr)
    matrix[2, 2] = cp * cr

    return matrix

def bbox_of_car(K, WorldToCameraRotation, car_id, find_poses, frame_cars, heightOffset, image_size, middle, scale):
    pos_car = frame_cars[car_id]['transform']  # Transformation this is the position of car world space.
    bbox_car = np.squeeze(
        np.asarray(np.matmul(np.matmul(WorldToCameraRotation, get_car_rotation_matrix(frame_cars[car_id]['yaw'])),
                             frame_cars[car_id]['bounding_box'])))  # Bbox rotated to the camera space
    point_0 = np.squeeze(np.asarray(np.matmul(WorldToCameraRotation, np.array(pos_car).reshape(
        (3, 1))) + middle))  # Que 5: Transformation -  This point is on relative to the camera space
    boxes_2d = []
    if find_poses:
        x_values = []
        y_values = []
        bbx = [np.array([point_0[0] - bbox_car[0], point_0[1] - bbox_car[1], point_0[2] - bbox_car[2]]),
               np.array([point_0[0] + bbox_car[0], point_0[1] - bbox_car[1], point_0[2] - bbox_car[2]]),
               np.array([point_0[0] + bbox_car[0], point_0[1] - bbox_car[1], point_0[2] + bbox_car[2]]),
               np.array([point_0[0] + bbox_car[0], point_0[1] + bbox_car[1], point_0[2] - bbox_car[2]]),
               np.array([point_0[0] + bbox_car[0], point_0[1] + bbox_car[1], point_0[2] + bbox_car[2]]),
               np.array([point_0[0] - bbox_car[0], point_0[1] - bbox_car[1], point_0[2] + bbox_car[2]]),
               np.array([point_0[0] - bbox_car[0], point_0[1] + bbox_car[1], point_0[2] - bbox_car[2]]),
               np.array([point_0[0] - bbox_car[0], point_0[1] + bbox_car[1], point_0[2] + bbox_car[2]])]
        for point in bbx:
            point_2D = np.matmul(K, point.reshape((3, 1)))
            point_2D = point_2D / point_2D[2]
            x_2d = int(image_size[0] - point_2D[0])
            y_2d = int(image_size[1] - point_2D[1])
            x_values.append(x_2d)
            y_values.append(y_2d)
        boxes_2d.append([y_values, x_values])
    point_0 = np.squeeze(np.asarray(np.matmul(WorldToCameraRotation, np.array(pos_car).reshape((3, 1))) + middle))
    bbox_car = bbox_car * scale
    point_0 = point_0 * scale
    bbox_car = bbox_car * [-1, -1, 1]  # normal camera coodrinates
    point_0 = point_0 * [-1, -1, 1]  # normal camera coodrinates
    point_0 = np.squeeze(np.asarray(np.matmul(P_inv, point_0)))
    point_0[2] = point_0[2] - heightOffset
    bbox_car = np.squeeze(np.asarray(np.matmul(P_inv, bbox_car)))
    car = np.column_stack((point_0 - np.abs(bbox_car), point_0 + np.abs(bbox_car)))
    return car, boxes_2d

def bbox_of_entity_alreadyInGlobalSpace(K, WorldToCameraRotation, car_id, find_poses, frame_cars, heightOffset,
                                        image_size, middle, scale, peopleFlag):
    minMaxWorldPos = frame_cars[car_id]['BBMinMax']
    assert minMaxWorldPos.shape == (3, 2), "Incorrect format should be xmin,xmax first row, then ymin, ymax etc"
    # scale down from meters to voxels
    minMaxWorldPos *= scale
    # Add height offset
    minMaxWorldPos[2, :] -= heightOffset
    return minMaxWorldPos, []

def get_cars_map(cars_dict, frames, R, middle, height, scale, K, filepath, people_flag=False, find_poses=False,
                 datasetOptions=None):
    image_sz = [800, 600]
    poses_map = {}
    init_frames = {}  # Map from car_id to first frame index when it appears in the scene
    valid_ids = []  # Cars that appear in frame 0
    cars_map = {}  # Map from car_id to a list of ordered (by frame) positions on each frame
    for i, frame in enumerate(frames):
        frame_cars = cars_dict[frame]
        for car_id in frame_cars:
            if datasetOptions is None or datasetOptions.dataAlreadyInWorldSpace is False:
                car, bboxes_2d = bbox_of_car(K, R, car_id, find_poses, frame_cars, height, image_sz, middle, scale)
            else:
                car, bboxes_2d = bbox_of_entity_alreadyInGlobalSpace(K, R, car_id, find_poses, frame_cars, height,
                                                                     image_sz, middle, scale, people_flag)

            if frame == 0:
                valid_ids.append(car_id)
            if car_id not in init_frames.keys():
                init_frames[car_id] = int(frame)
            if not people_flag:
                car = car.flatten()

            if not car_id in cars_map:
                cars_map[car_id] = []
            cars_map[car_id].append(car)  # .astype(int))
    return cars_map, poses_map, valid_ids, init_frames

def get_people_and_cars(WorldToCameraRotation, cars, filepath, frames, WorldToCameraTranslation, heightOffset,
                            people, scale_x, find_poses=False, datasetOptions=None):
    # TODO: make this an utility, do not leave it here
    simplifyDataSet = False  # Used for debugging purposes to cut the load time of cars/people stuff

    # Read the intrinsics matrix if needed
    K = None
    if datasetOptions is None or datasetOptions.dataAlreadyInWorldSpace is False:
        K = get_camera_intrinsic_matrix(filepath)

    # Read the cars and people dictionary in format: {frame_id : { entity_id : {...dict with data... } } }
    cars_path = os.path.join(filepath, FILENAME_CARS_TRAJECTORIES)
    with open(cars_path, 'rb') as handle:
        cars_dict = pickle.load(handle)

    if simplifyDataSet == True and len(cars_dict) > 1:
        cars_dict_new = {}
        for x in range(len(frames)):
            cars_dict_new[x] = cars_dict[x]
        cars_dict = cars_dict_new
        with open(cars_path, 'wb') as handle:
            pickle.dump(cars_dict, handle, protocol=2) # Protocol 2 ensures compatibility between python 2 and 3

    people_path = os.path.join(filepath, FILENAME_PEOPLE_TRAJECTORIES)
    with open(people_path, 'rb') as handle:
        people_dict = pickle.load(handle)

    if simplifyDataSet == True and len(people_dict) > 1:
        people_dict_new = {}
        for x in range(len(frames)):
            if x in people_dict:
                people_dict_new[x] = people_dict[x]
            else:
                people_dict_new[x] = {}

        people_dict = people_dict_new
        with open(people_path, 'wb') as handle:
            pickle.dump(people_dict, handle, protocol=2)# Protocol 2 ensures compatibility between python 2 and 3

    # see the comments below to understand what these means
    cars, cars_2D = get_cars_list(cars, cars_dict, frames, WorldToCameraRotation, WorldToCameraTranslation,
                                  heightOffset, scale_x, K, filepath, find_poses=find_poses,
                                  datasetOptions=datasetOptions)
    people, people_2D = get_cars_list(people, people_dict, frames, WorldToCameraRotation, WorldToCameraTranslation,
                                      heightOffset, scale_x, K, filepath, people_flag=True, find_poses=find_poses,
                                      datasetOptions=datasetOptions)
    # print people

    # init_frames = frame id when each car_id and pedestrian_id appears for the first time in the scene
    # poses, cars 2D, people2D  = 2d stuff - not working now..
    # ped_dict = {pedestrian_id -> list of positions ordered by frame | positions are [x,y,z] space in voxels, world coordinate}
    # cars_dict = {car_id -> list of bbox ordered by frame | bboxes are ooab in voxels, [xmin, xmax+1, ymin, ymax+1, zmin, zmax+1] world coordinate}
    # cars, people have the same coordinates and bboxes as defined above that are organized differently:
    #       they lists of lists such [frame 0 ; frame 1 ; ... frame num frames], where each frame data is a list [ position / ooab for each pedestrian / car index]
    #                           note that info is indexed by the sorted keys of pedestrian / car_id !].
    ped_dict, poses, valid_ids, init_frames = get_cars_map(people_dict, frames, WorldToCameraRotation,
                                                           WorldToCameraTranslation, heightOffset, scale_x, K, filepath,
                                                           people_flag=True, datasetOptions=datasetOptions)

    car_dict, _, _, init_frames_cars = get_cars_map(cars_dict, frames, WorldToCameraRotation, WorldToCameraTranslation,
                                                    heightOffset, scale_x, K, filepath, people_flag=False,
                                                    datasetOptions=datasetOptions)
    return cars, people, ped_dict, cars_2D, people_2D, poses, valid_ids, car_dict, init_frames, init_frames_cars


# Reads the point cloud from a single frame (segmented labelels and rgb ones) and append those to the input/output dictionaries
# plyfile - is the frame with point cloud segmented data path
def combine_frames(WorldToCameraRotation, filepath, WorldToCameraTranslation,
                   middle_height, plyfile, reconstruction_label, reconstruction_rgb,
                   scale, datasetOptions=None):
    filename = os.path.basename(plyfile)
    print ("Processing file " , plyfile)
    pointcloud_3D = read_3D_pointcloud(plyfile, '')  # THis is the semantic labeled point cloud

    # The file without _seg contains the RGB segmentation labels of each 3d point
    parts = filename.split('_seg')
    rgb_dict = get_rgb_point_cloud(filepath,
                                   parts)  # This is the RGB labels, rgb_dict{(x,y,z)}->(R,G,B) segmentation label

    for point in pointcloud_3D:
        # Each 3D position in the dictionary contains at most 3 labels voted for this frame.
        # Take the one that is the most voted
        pos = (point[0], point[1], point[2])
        labels = [point[3]] if len(point) == 4 else [point[3], point[4], point[5]]
        counts = np.bincount(labels)
        label = np.argmax(counts)  # voted label

        # If car or pedestrian, remove from reconstruction

        if not isLabelIgnoredInReconstruction(labels):
            point_new_coord = np.array(pos)

            # Does it needs to conversion or is it already in global space ?
            if datasetOptions is None or datasetOptions.dataAlreadyInWorldSpace is False:
                point_new_coord = np.squeeze(
                    np.asarray(np.matmul(WorldToCameraRotation, np.array(pos).reshape((3, 1))) + WorldToCameraTranslation))
                point_new_coord = point_new_coord * [-1, -1, 1]  # In camera coordinate system  of cityscapes
                point_new_coord = np.matmul(P_inv, point_new_coord)  # Now the point is in vehicle space, citiyscapes

            # Scale the point as needed to our framework environment
            point_new_coord = tuple(np.round(point_new_coord * scale).astype(int))

            # If road or sidewalks, just add the coordinate to the heights statistics
            # The second condition ensures that only the point data very close to left/right on our sensor are taken into account. There is a lot of noise around...
            if isCloudPointRoadOrSidewalk(pos, label): #and (point_new_coord[1] < (1 * scale)):
                middle_height.append(point_new_coord[2])


            # If  point not in dict, add it.
            if point_new_coord not in reconstruction_label:
                reconstruction_label[point_new_coord] = []
                reconstruction_rgb[point_new_coord] = []
            else:
                # If the previous points had only unsefull points, erase all
                # We add only a single unsefull point that's why we can put this condition easly
                if isUsefulLabelForReconstruction(label) and len(reconstruction_label[point_new_coord]) == 1 and (not isUsefulLabelForReconstruction(reconstruction_label[point_new_coord][0])):
                    reconstruction_label[point_new_coord].clear()

            # Add point only if it is other than no label, or it is the only one available
            if isUsefulLabelForReconstruction(label) or len(reconstruction_label[point_new_coord]) == 0:
                reconstruction_label[point_new_coord].append(label)
                reconstruction_rgb[point_new_coord].append(rgb_dict[pos]) # Set the RGB segmentation value


''' 
REconstructs or read one already existing in filepath if recalculate is False
local_scale_x - is the scale to move from the source data system to our framework (voxels, which in this moment 1m = 5 voxels, each having 20 cm)
label_mapping - must contain a table that maps from source segmentation labels to citiscapes segmentation labels (which is the reference ones). 
find_poses - if you need 2D labels
datasetOptions - various hacks to inject your own options from the source dataset...
'''
def reconstruct3D_ply(filepath, local_scale_x=5, recalculate=False, find_poses=False, read_3D=True,
                      datasetOptions=None):
    frameSkip = 1 # How many frames modulo to skip for reconstruction (if needed. default is 50)
    numInitialFramesToIgnore = 0 # How many frames in the beginning of the scene to ignore

    FRAMEINDEX_MIN, FRAMEINDEX_MAX = None, None
    if datasetOptions is None or datasetOptions.dataAlreadyInWorldSpace is False:
        WorldToCameraRotation, WorldToCameraTranslation, frames = get_camera_matrix(filepath)
        frameSkip = 50
        numInitialFramesToIgnore = 1
    else:
        WorldToCameraRotation, WorldToCameraTranslation, frames = np.eye(3), np.zeros(shape=(3,1)), datasetOptions.framesIndicesCaptured
        frameSkip = datasetOptions.frameSkip
        numInitialFramesToIgnore = datasetOptions.numInitialFramesToIgnore

    FRAMEINDEX_MIN = int(min(frames))
    FRAMEINDEX_MAX = int(max(frames))

    people = []
    cars = []
    for frame in range(min(0, FRAMEINDEX_MIN), (FRAMEINDEX_MAX + 1)):
        cars.append([])
        people.append([])

    # Try read point cloud
    reconstruction_path             = os.path.join(filepath, FILENAME_COMBINED_CARLA_ENV_POINTCLOUD)
    reconstruction_path_segColored  = os.path.join(filepath, FILENAME_COMBINED_CARLA_ENV_POINTCLOUD_SEGCOLOR)
    scale = local_scale_x
    labels_mapping = get_label_mapping()

    centeringFilePath = os.path.join(filepath, FILENAME_CENTERING_ENV)

    if read_3D:
        # If reconstruction files are saved on disk already, and not forcing a recalculation, reload it
        # Centering file is a dumped dictionary contains 3 values: the hight offset, middle - world to camera translation, scale - scale value for converting from dataspace to voxel space
        if os.path.isfile(reconstruction_path) and os.path.isfile(centeringFilePath) and not recalculate:
            print ("Loading existing reconstruction")

            with open(centeringFilePath, 'rb') as centeringFileHandle:
                centering = pickle.load(centeringFileHandle)
                scale = centering['scale']

            plydata = PlyData.read(reconstruction_path)
            nr_points = plydata.elements[0].count
            pointcloud_3D = np.array([plydata['vertex'][k] for k in range(nr_points)])
            final_reconstruction = {}  # final_reconstruction[(x,y,z)] = (rgb label, label id from segmentation)
            for point in pointcloud_3D:
                final_reconstruction[(int(point[0]), int(point[1]), int(point[2]))] = (
                point[3], point[4], point[5], point[6])
        else:
            print ("Create reconstruction")

            # For each 3D (x,y,z) point, which is the rgb and label (segmentation) in source domain ?
            reconstruction_rgb = {}
            reconstruction_label = {}

            # Collect Middle Heights of the analized segmentation frame
            middle_height = []

            # A small hack to force frame 0 even if not specified...it is usefull because sometimes this represent the startup position of the reference point and we might want to get statstics out of that
            isFrame_0_needed = (datasetOptions is not None and datasetOptions.dataAlreadyInWorldSpace == True)

            # Take all *seg.ply files (basically different frames along the scene) in the filepath and combine the information from them in the hashes created above
            # These files contains the segmentated 3D point cloud to build the maps above.
            for plyfile in sorted(glob.glob(os.path.join(filepath, '0*_seg.ply'))):
                frame_nbr = os.path.basename(plyfile)
                frame_nbr = int(frame_nbr[:-len("_seg.ply")])
                # frames.append(frame_nbr)

                if (FRAMEINDEX_MIN is not None and frame_nbr < FRAMEINDEX_MIN) and (not isFrame_0_needed):
                    continue
                if FRAMEINDEX_MAX is not None and frame_nbr > FRAMEINDEX_MAX:
                    break

                if (frame_nbr >= numInitialFramesToIgnore and frame_nbr % frameSkip == 0) or (frame_nbr == 0 and isFrame_0_needed):
                    combine_frames(WorldToCameraRotation, filepath, WorldToCameraTranslation, middle_height, plyfile,
                                   reconstruction_label, reconstruction_rgb, scale, datasetOptions)

            # Offset everything by the minimum of heights in the scene (height normalization)
            height = np.min(middle_height)

            # Take all 3D point cloud votes and select the mode for each one.
            combined_reconstruction = []
            final_reconstruction = {}
            for point in reconstruction_rgb:
                assert len(reconstruction_rgb[point]) > 0

                rgb = max(set(reconstruction_rgb[point]), key=reconstruction_rgb[point].count)
                label = max(set(reconstruction_label[point]), key=reconstruction_label[point].count)
                combined_reconstruction.append(
                    (point[0], point[1], point[2] - height, rgb[0], rgb[1], rgb[2], labels_mapping[label]))
                final_reconstruction[(int(point[0]), int(point[1]), int(point[2] - height))] = (rgb[0], rgb[1], rgb[2],
                                                                                                labels_mapping[
                                                                                                    label])  # final_reconstruction[(x,y,z)] = (rgb label, label id from segmentation)

            # Save the segmentation reconstruction file in the same folder for caching purposes
            save_3d_pointcloud(combined_reconstruction, reconstruction_path)
            save_3d_pointcloud_asSegLabel(combined_reconstruction, reconstruction_path_segColored)

            allPointsCoords = np.array(list(final_reconstruction.keys()))
            minBBox = allPointsCoords.min(axis = 0)
            maxBBox = allPointsCoords.max(axis = 0)

            # Save the centering file
            centering = {}
            centering['height'] = height
            centering['middle'] = WorldToCameraTranslation
            centering['scale'] = local_scale_x
            centering['frame_min'] = FRAMEINDEX_MIN
            centering['frame_max'] = FRAMEINDEX_MAX
            centering['min_bbox'] = list(minBBox)
            centering['max_bbox'] = list(maxBBox)

            with open(centeringFilePath, 'wb') as centeringFileHandle:
                pickle.dump(centering, centeringFileHandle, protocol=2) # Protocol 2 ensures compatibility between python 2 and 3
    else:
        centering = {}
        with open(centeringFilePath, 'rb') as centeringFileHandle:
            centering = pickle.load(centeringFileHandle)

        plydata = PlyData.read(reconstruction_path)
        nr_points = plydata.elements[0].count
        pointcloud_3D = np.array([plydata['vertex'][k] for k in range(nr_points)])

        # Note : the height (Z axis) offset is already offset in the reconstruction
        final_reconstruction = {}
        for point in pointcloud_3D:
            final_reconstruction[(int(point[0]), int(point[1]), int(point[2]))] = (
            point[3], point[4], point[5], point[6])

    # See the comments inside the called function, before return to understand meaning of all these
    cars, people, ped_dict, cars_2D, people_2D, poses, valid_ids, car_dict, init_frames, init_frames_cars = get_people_and_cars(
            WorldToCameraRotation,
            cars,
            filepath,
            frames,
            centering['middle'],
            centering['height'],
            people,
            centering['scale'],
            find_poses=find_poses,
            datasetOptions=datasetOptions)
    if find_poses:
        save_2D_poses(filepath, people_2D, frames)

    return final_reconstruction, people, cars, local_scale_x, ped_dict, cars_2D, people_2D, valid_ids, car_dict, init_frames, init_frames_cars


# TODO: create a factory here maybe in the future to handle different datasets...
def CreateDefaultDatasetOptions_Waymo(metadata):
    class DatasetOptions:
        def __init__(self):
            self.dataAlreadyInWorldSpace = True  # Is data already converted to world space ?
            self.framesIndicesCaptured = list(np.arange(min(0, metadata['frame_min']), metadata['frame_max']))
            self.LIMIT_FRAME_NUMBER = metadata['frame_max']  # How much we want to go with processing...max is default
            self.frameSkip = 1  # Not needed on inference or training, just for reconstruction process
            self.numInitialFramesToIgnore = (metadata['frame_min'] - 1)  # Somehow redundant, but it is used in CARLA for example to force ignore first frame...

    options = DatasetOptions()
    return options
