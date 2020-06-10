# Based on previously segmented data, and waymo 3D points data, reconstruct the point cloud
# Save two files for each frame: frame_number.ply and frame_number_seg.ply.
# The former contains the poincloud with RGB, the second with segmentation only (in CARLA dataset space !!!)

import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
from pipeline_commons import *
import collections

tf.enable_eager_execution()
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from plyfile import PlyData, PlyElement
from scipy import stats
from PIL import Image
import pickle

# Some internal settings
#--------------------------------
IS_DEBUG_ENABLED = False
DEBUG_FRAMEINDEX_LIMIT = 30 # If should stop after first frame or not...good for fast debugging
IS_RANGE_COLOR_ShOWING_ENABLED = False # DEbug to see range projections on images plot
returnsIndicesToUse = [0, 1] # Should we use just first laser return or both ?
cameraindices_to_use = [0, 1, 2, 3, 4] # The camera indices to project on
DEBUG_SEGMENTATION_ENABLED = True # If True, will dump input.output of images from segmentation
NO_CAMERA_INDEX = -1
NO_LABEL_POINT = 0

Point3DInfoType = collections.namedtuple('Point3DInfo', ['x', 'y', 'z', 'R', 'G', 'B', 'segLabel', 'segR', 'segG', 'segB'])

# If true, the output of the point cloud will be in the world space output, relative to the reference point (first pose of the vehicle in world)
DO_OUTPUT_IN_WORLD_SPACE = True

ade20KToCarla = None # A dictionary containing mapping from [ADE20K label - > CARLA label] for mapping the output segmentation values
#--------------------------------

# Some utility functions
#---------------------------------
def show_camera_image(frame, camera_image, camera_labels, layout, cmap=None):
  """Show a camera image and the given camera labels."""

  ax = plt.subplot(*layout)

  # Draw the camera labels.
  for camera_labels in frame.camera_labels:
    # Ignore camera labels that do not correspond to this camera.
    if camera_labels.name != camera_image.name:
      continue

    # Iterate over the individual labels.
    for label in camera_labels.labels:
      # Draw the object bounding box.
      ax.add_patch(patches.Rectangle(
        xy=(label.box.center_x - 0.5 * label.box.length,
            label.box.center_y - 0.5 * label.box.width),
        width=label.box.length,
        height=label.box.width,
        linewidth=1,
        edgecolor='red',
        facecolor='none'))

  # Show the camera image.
  imgDecoded = tf.image.decode_jpeg(camera_image.image)
  plt.imshow(imgDecoded, cmap=cmap)
  plt.title(open_dataset.CameraName.Name.Name(camera_image.name))
  plt.grid(False)
  plt.axis('off')

def plot_range_image_helper(data, name, layout, vmin = 0, vmax=1, cmap='gray'):
  """Plots range image.

  Args:
    data: range image data
    name: the image title
    layout: plt layout
    vmin: minimum value of the passed data
    vmax: maximum value of the passed data
    cmap: color map
  """
  plt.subplot(*layout)
  plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
  plt.title(name)
  plt.grid(False)
  plt.axis('off')

def get_range_image(range_images, laser_name, return_index):
  """Returns range image given a laser name and its return index."""
  return range_images[laser_name][return_index]

def show_range_image(range_image, layout_index_start = 1):
  """Shows range image.

  Args:
    range_image: the range image data from a given lidar of type MatrixFloat.
    layout_index_start: layout offset
  """
  range_image_tensor = tf.convert_to_tensor(range_image.data)
  range_image_tensor = tf.reshape(range_image_tensor, range_image.shape.dims)
  lidar_image_mask = tf.greater_equal(range_image_tensor, 0)
  range_image_tensor = tf.where(lidar_image_mask, range_image_tensor,
                                tf.ones_like(range_image_tensor) * 1e10)
  range_image_range = range_image_tensor[...,0]
  range_image_intensity = range_image_tensor[...,1]
  range_image_elongation = range_image_tensor[...,2]
  plot_range_image_helper(range_image_range.numpy(), 'range',
                   [8, 1, layout_index_start], vmax=75, cmap='gray')
  plot_range_image_helper(range_image_intensity.numpy(), 'intensity',
                   [8, 1, layout_index_start + 1], vmax=1.5, cmap='gray')
  plot_range_image_helper(range_image_elongation.numpy(), 'elongation',
                   [8, 1, layout_index_start + 2], vmax=1.5, cmap='gray')

def rgba(r):
  """Generates a color based on range.

  Args:
    r: the range value of a given point.
  Returns:
    The color for a given range
  """
  c = plt.get_cmap('jet')((r % 20.0) / 20.0)
  c = list(c)
  c[-1] = 0.5  # alpha
  return c

def plot_image(camera_image):
  """Plot a cmaera image."""
  plt.figure(figsize=(20, 12))
  plt.imshow(tf.image.decode_jpeg(camera_image.image))
  plt.grid("off")

def plot_points_on_image(projected_points, camera_image, rgba_func,
                         point_size=5.0):
  """Plots points on a camera image.

  Args:
    projected_points: [N, 3] numpy array. The inner dims are
      [camera_x, camera_y, range].
    camera_image: jpeg encoded camera image.
    rgba_func: a function that generates a color from a range value.
    point_size: the point size.

  """
  plot_image(camera_image)

  xs = []
  ys = []
  colors = []

  for point in projected_points:
    xs.append(point[0])  # width, col
    ys.append(point[1])  # height, row
    colors.append(rgba_func(point[2]))

  plt.scatter(xs, ys, c=colors, s=point_size, edgecolors="none")


# These two functions read the the RGB and segmentation data corresponding to one frame.
# The output is {'camera index' : data } where data is [Height, Width, RGB] array
def readRGBDataForFrame(frameIndex, segmentName):
    RGB_IMAGES_PATH = os.path.join(SEG_INPUT_IMAGES_BASEPATH, segmentName, SEG_INPUT_IMAGES_RGBFOLDER)

    outData = {}
    for cameraIndex in cameraindices_to_use:
        pic = Image.open(os.path.join(RGB_IMAGES_PATH, f"frame_{frameIndex}_img_{cameraIndex}.jpg"))
        outData[cameraIndex] = np.array(pic)
    return outData

def readSEGDataForFrame(frameIndex, segmentName):
    SEG_IMAGES_PATH = os.path.join(SEG_OUTPUT_LABELS_BASEFILEPATH, segmentName, SEG_OUTPUT_LABELS_SEGFOLDER)
    targetFilePath = os.path.join(SEG_IMAGES_PATH, f"labels_{frameIndex}.pkl")
    outData = pickle.load(open(targetFilePath, "rb"))

    assert sorted(outData.keys()) == sorted(cameraindices_to_use), "Invalid data content !"
    return outData

def save_3d_pointcloud_asSegLabel(points_3d, filename):
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
                  'property uchar label',
                  'end_header']
        return '\n'.join(header).format(points)

    for point in points_3d:
        assert len(point) == 10, "invalid data input" # xyz rgb label
        for p in point:
            try:
                n = float(p)
            except ValueError:
                print ("Problem " + str(point))
    # points_3d = np.concatenate(
    #     (point_list._array, self._color_array), axis=1)

    try:
        ply = '\n'.join(
            ['{:.2f} {:.2f} {:.2f} {:.0f}'.format(p.x, p.y, p.z, p.segLabel) for p in points_3d])  # .replace('.',',')
    except ValueError:
        for point in points_3d:
            print (point)
            print ('{:.2f} {:.2f} {:.2f}{:.0f}'.format(*point))
    # Create folder to save if does not exist.
    # folder = os.path.dirname(filename)
    # if not os.path.isdir(folder):
    #     os.makedirs(folder)

    # Open the file and save with the specific PLY format.
    with open(filename, 'w+') as ply_file:
        ply_file.write('\n'.join([construct_ply_header(), ply]))

def save_3d_pointcloud_asSegColored(points_3d, filename):
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
                  'end_header']
        return '\n'.join(header).format(points)

    for point in points_3d:
        assert len(point) == 10, "invalid data input" # xyz rgb label
        for p in point:
            try:
                n = float(p)
            except ValueError:
                print ("Problem " + str(point))
    # points_3d = np.concatenate(
    #     (point_list._array, self._color_array), axis=1)
    try:
        ply = '\n'.join(
            ['{:.2f} {:.2f} {:.2f} {:.0f} {:.0f} {:.0f}'.format(p.x, p.y, p.z, p.segR, p.segG, p.segB) for p in points_3d])  # .replace('.',',')
    except ValueError:
        for point in points_3d:
            print (point)
            print ('{:.2f} {:.2f} {:.2f} {:.0f} {:.0f} {:.0f}'.format(*point))
    # Create folder to save if does not exist.
    # folder = os.path.dirname(filename)
    # if not os.path.isdir(folder):
    #     os.makedirs(folder)

    # Open the file and save with the specific PLY format.
    with open(filename, 'w+') as ply_file:
        ply_file.write('\n'.join([construct_ply_header(), ply]))

def save_3d_pointcloud_asRGB(points_3d, filename):
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
                  'end_header']
        return '\n'.join(header).format(points)

    for point in points_3d:
        assert len(point) == 10, "invalid data input"  # xyz rgb label
        for p in point:
            try:
                n = float(p)
            except ValueError:
                print("Problem " + str(point))
    # points_3d = np.concatenate(
    #     (point_list._array, self._color_array), axis=1)
    try:
        ply = '\n'.join(
            ['{:.2f} {:.2f} {:.2f} {:.0f} {:.0f} {:.0f}'.format(p.x, p.y, p.z, p.R, p.G, p.B) for p in
             points_3d])  # .replace('.',',')
    except ValueError:
        for point in points_3d:
            print(point)
            print('{:.2f} {:.2f} {:.2f} {:.0f} {:.0f} {:.0f}'.format(*point))
    # Create folder to save if does not exist.
    # folder = os.path.dirname(filename)
    # if not os.path.isdir(folder):
    #     os.makedirs(folder)

    # Open the file and save with the specific PLY format.
    with open(filename, 'w+') as ply_file:
        ply_file.write('\n'.join([construct_ply_header(), ply]))

def read_3D_pointcloud(filename, file_end='/dense/fused_text.ply'):
    plydata_3Dmodel = PlyData.read(filename + file_end)
    nr_points = plydata_3Dmodel.elements[0].count
    pointcloud_3D = np.array([plydata_3Dmodel['vertex'][k] for k in range(nr_points)])
    return pointcloud_3D

# Aggregates all points given in the output dictionary, for each 3D point cloud will add the R G B in the scene and the segmentation label
# unprojectedPoints is True if the points have no label or camera projection image
def processPoints(points3D_and_cp, outPlyDataPoints, imageCameraIndex = NO_CAMERA_INDEX, rgbData = None, segData = None):
    for point in points3D_and_cp:
        x,y,z = point[0:3]
        key = (x, y, z)

        camX, camY = None, None
        R, G, B = 0, 0, 0
        label = NO_LABEL_POINT

        if imageCameraIndex != NO_CAMERA_INDEX:
            rgbCamera = rgbData[imageCameraIndex]
            segCamera = segData[imageCameraIndex]

            camX, camY = int(point[3]), int(point[4])

#            try:
            R, G, B = rgbCamera[camY, camX, :]
            label = int(segCamera[camY, camX]) + 1 # WHY DO WE ADD + 1 here ?? BECAUSE ADE20K returns in range [0,149] while naming and all dictionaries are in [1,150]

            label = ade20KToCarla[label] # Move to CARLA segmentation values
#            except:
#                print("Exception")
#                pass

        if key not in outPlyDataPoints:
            outPlyDataPoints[key] = []

        outPlyDataPoints[key].append((R, G, B, label))

# Takes the output dictionary build as above by ProcessPoints method and returns a flattened list
def convertDictPointsToList(inPlyDataPoints):
    outPlyDataPoints = []

    stat_numPointsWhereVotesAreNeeded = 0 # just for some stat
    for keyPoint3D, pointData in inPlyDataPoints.items():
        x, y, z = keyPoint3D[0], keyPoint3D[1], keyPoint3D[2]

        # Select the mode label from the point list
        pointData = np.array(pointData).reshape(-1, 4) # R,G,B, seg label
        mode, count = stats.mode(pointData[:, 3])
        votedLabel = mode[0]

        #COMMENT THIS - Debug to see multiple points on the same coordinate:
        if pointData.shape[0] > 1:
            #print("P: ({:.2f} {:.2f} {:.2f}) {}".format(x, y, z, pointData))
            stat_numPointsWhereVotesAreNeeded += 1

        # end debug

        votedRGB = None
        # Take first RGB corresponding to the mode
        for dataIndex in range(pointData.shape[0]):
            if pointData[dataIndex, 3] == votedLabel:
                votedRGB = (x, y, z, pointData[dataIndex, 0], pointData[dataIndex, 1], pointData[dataIndex, 2])
                break

        segColor = carla_label_colours[votedLabel]
        outPlyDataPoints.append(Point3DInfoType(*votedRGB, votedLabel, *segColor))

    print(f"Needed votes from {stat_numPointsWhereVotesAreNeeded} out of {len(inPlyDataPoints)}")
    return outPlyDataPoints
#-----------------------------------------------

def getPointCloudPointsAndCameraProjections(frame, thisFramePose, worldToReferencePointTransform):
    # Get the range images, camera projects from the frame data
    (range_images, camera_projections, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

    if IS_DEBUG_ENABLED:
        # 2. Print some cameras images in the data
        plt.figure(figsize=(25, 20))
        for index, image in enumerate(frame.images):
            # test_object_detection(image.image)
            show_camera_image(frame, image, frame.camera_labels, [3, 3, index + 1])

        # 3. Print range images
        plt.figure(figsize=(64, 20))
        frame.lasers.sort(key=lambda laser: laser.name)
        show_range_image(get_range_image(range_images, open_dataset.LaserName.TOP, 0), 1)
        show_range_image(get_range_image(range_images, open_dataset.LaserName.TOP, 1), 4)

    # Step 2: Convert images to point cloud and view their content
    # points = Num lasers * [ [x, y, z In Vehicle frame] * num_points_with_range > 0]
    # cp_points = Num lasers *  [ [name Of camera 1, x in cam 1, y in cam 1], ...same for cam 2]
    # The indices and shapes are them same => we can take points[P] and say that its coordinates in vehicle frame
    # corresponds to cp_points[P] -> giving us the camera name of the projections plus the x,y in the respective camera.
    # My question is why there are two camera projections but probably this is another story...
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose)

    # Same as above but for return index 1
    points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose,
        ri_index=1)

    # 3d points in vehicle frame.
    points_all_ri0 = np.concatenate(points, axis=0)
    points_all_ri2 = np.concatenate(points_ri2, axis=0)
    # camera projection corresponding to each point.
    cp_points_all_ri0 = np.concatenate(cp_points, axis=0)
    cp_points_all_ri2 = np.concatenate(cp_points_ri2, axis=0)

    # These are the 3d points and camera projection data for each
    points_byreturn     = [points_all_ri0, points_all_ri2]
    cp_points_byreturn  = [cp_points_all_ri0, cp_points_all_ri2]


    # Last step: convert all points from vehicle frame space to world space - relative to the first vehicle frame pose
    # Get the vehicle transform to world in this frame
    if DO_OUTPUT_IN_WORLD_SPACE is True:
        vehicleToWorldTransform     = thisFramePose
        vehicleToWorldRotation      = vehicleToWorldTransform[0:3, 0:3]
        vehicleToWorldTranslation   = vehicleToWorldTransform[0:3, 3]

        # Then world transform back to reference point
        worldToReferenceRotation    = worldToReferencePointTransform[0:3, 0:3]
        worldToReferenceTranslation = worldToReferencePointTransform[0:3, 3]

        #
        for returnIndex, allPoints in enumerate(points_byreturn):
            # Transform all points to world coordinates first
            allPoints   = tf.einsum('jk,ik->ij', vehicleToWorldRotation, allPoints) + tf.expand_dims(vehicleToWorldTranslation, axis=-2)

            # Then from world to world reference
            allPoints     = tf.einsum('jk,ik->ij', worldToReferenceRotation, allPoints) + tf.expand_dims(worldToReferenceTranslation, axis=-2)

            # Copy back
            points_byreturn[returnIndex] = allPoints.numpy()

    return points_byreturn, cp_points_byreturn


def doPointCloudReconstruction(recordSegmentFiles):

    # These are the transformation from world frame to the reference vehicle frame (first pose of the vehicle in the world)
    worldToReferencePointTransform = None

    # Iterate over all segmentation names sent
    for segmentPath in recordSegmentFiles:
        assert os.path.exists(segmentPath), (f'The file you specified {segmentPath} doesn\'t exist !')

        segmentName = extractSegmentNameFromPath(segmentPath)
        segmentedDataPath = os.path.join(SEG_OUTPUT_LABELS_BASEFILEPATH, segmentName, SEG_OUTPUT_LABELS_SEGFOLDER)

        # 1. Iterate over frame by frame of a segment
        dataset = tf.data.TFRecordDataset(segmentPath, compression_type='')
        for frameIndex, data in enumerate(dataset):
            print(f"Extracting point cloud from frame {frameIndex}...")
            # Step 1: Read the frame in bytes
            # -------------------------------------------
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            # All point cloud data is relative to the car with sensors attached
            # We can transform to world positions using frame.pose.transform.
            # But to get coordinates back to the reference frame - WHICH IS THE INITIAL POSITION OF THE CAR - we use the below inverse matrix
            frame_thisPose = np.reshape(np.array(frame.pose.transform).astype(np.float32), [4, 4])
            if frameIndex == 0:
                worldToReferencePointTransform  = np.linalg.inv(frame_thisPose)
            # -------------------------------------------


            # Step 2: Process the frame and get the 3d lidar points and camera projected points from the frame, for each return
            # -------------------------------------------
            points_byreturn, cp_points_byreturn = getPointCloudPointsAndCameraProjections(frame, frame_thisPose, worldToReferencePointTransform)
            # -------------------------------------------


            # Step 3: iterate over all point cloud data and camera projections and fill a dictionary with each 3D point data
            # Here we gather all ply data in format: {(x,y,z) : [r g b label]], for all points in the point cloud.
            # Why dictionary ? Because we might want to discretize from original space to a lower dimensional space so same x,y,z from original data might go into the same chunk
            # -------------------------------------------
            plyDataPoints = {}

            # Read the RGB images and segmentation labels corresponding for this frame
            RGB_Data = readRGBDataForFrame(frameIndex, segmentName)
            SEG_Data = readSEGDataForFrame(frameIndex, segmentName)

            # Sort images by key index
            images = sorted(frame.images, key=lambda i: i.name)

            # For each return
            for returnIndex in returnsIndicesToUse:
                # Put together [projected camera data, 3D points] on the same row, convert to tensor
                cp_points_all_concat = np.concatenate([cp_points_byreturn[returnIndex], points_byreturn[returnIndex]],
                                                      axis=-1)
                cp_points_all_concat_tensor = tf.constant(cp_points_all_concat)

                # Compute the distance between lidar points and vehicle frame origin for each 3d point
                distances_all_tensor = tf.norm(points_byreturn[returnIndex], axis=-1,
                                               keepdims=True) if IS_RANGE_COLOR_ShOWING_ENABLED is True else None

                # Create a tensor with all projected [projected camera data] and one with 3d points
                cp_points_all_tensor = tf.constant(cp_points_byreturn[returnIndex], dtype=tf.int32)
                points_3D_all_tensor = tf.constant(points_byreturn[returnIndex], dtype=tf.float32)

                # For each camera image index
                image_projections_indices = [0, 1]
                for imageProjIndex in image_projections_indices:
                    for image_index in cameraindices_to_use:
                        assert len(images) == len(cameraindices_to_use), ("looks like you need to add more data in image_projection_indices and see if I didn't do any hacks with it ! Waymo data changed")

                        # A mask with True where the camera projection points where on this image index
                        mask = tf.equal(cp_points_all_tensor[..., 0 if imageProjIndex == 0 else 3],
                                        images[image_index].name)

                        # Now select only camera projection data and 3D points in vehicle frame associated with this camera index
                        cp_points_all_tensor_camera = tf.cast(tf.gather_nd(cp_points_all_tensor, tf.where(mask)),
                                                              dtype=tf.float32)
                        points_3D_all_tensor_camera = tf.cast(tf.gather_nd(points_3D_all_tensor, tf.where(mask)),
                                                              dtype=tf.float32)

                        # Select only the cp points associated with the iterated camera projection index
                        cp_points_all_tensor_camera_byProjIndex = cp_points_all_tensor_camera[...,1:3] if imageProjIndex == 0 \
                                                                    else cp_points_all_tensor_camera[..., 4:6]

                        # Associate on each row one 3D point with its corresponding image camera projection (this index being iterated on)
                        # We have now on each row:  [(x,y,z, camX, camY)]
                        points3D_and_cp = tf.concat([points_3D_all_tensor_camera, cp_points_all_tensor_camera_byProjIndex], axis=-1).numpy()

                        # Gather these points in the output dictionary
                        processPoints(points3D_and_cp, plyDataPoints, imageCameraIndex=image_index, rgbData = RGB_Data, segData = SEG_Data)

                        # Demo showing...
                        if IS_RANGE_COLOR_ShOWING_ENABLED:
                            distances_all_tensor_camera = tf.gather_nd(distances_all_tensor, tf.where(mask))

                            # Associate on each row, with each x,y in camera space, the distance from vehicle frame to the 3D point lidar
                            projected_points_all_from_raw_data = tf.concat(
                                [cp_points_all_tensor_camera[..., 1:3],
                                 distances_all_tensor_camera] if imageProjIndex == 0 else \
                                    [cp_points_all_tensor_camera[..., 4:6], distances_all_tensor_camera],
                                axis=-1).numpy()

                            plot_points_on_image(projected_points_all_from_raw_data, images[image_index], rgba, point_size=5.0)

                # Add all point cloud points which are not not labeled (not found in a projected camera image)
                mask = tf.equal(cp_points_all_tensor[..., 0], 0)  # 0 is the index for unprojected camera point, on first cp index
                points_3D_all_unprojected_tensor = tf.cast(tf.gather_nd(points_3D_all_tensor, tf.where(mask)), dtype=tf.float32)
                processPoints(points_3D_all_unprojected_tensor.numpy(), plyDataPoints, imageCameraIndex=NO_CAMERA_INDEX)

            plyDataPointsFlattened = convertDictPointsToList(plyDataPoints)

            folderOutput    = os.path.join(POINTCLOUD_OUTPUT_BASEFILEPATH, segmentName)
            outputFramePath_rgb =  os.path.join(folderOutput, ("{0:05d}.ply").format(frameIndex))
            outputFramePath_seg =  os.path.join(folderOutput, ("{0:05d}_seg.ply").format(frameIndex))
            outputFramePath_segColored = os.path.join(folderOutput, ("{0:05d}_segColor.ply").format(frameIndex))

            save_3d_pointcloud_asRGB(plyDataPointsFlattened, outputFramePath_rgb)  # TODO: save one file with test_x.ply, using RGB values, another one test_x_seg.ply using segmented data.
            save_3d_pointcloud_asSegLabel(plyDataPointsFlattened, outputFramePath_seg)
            save_3d_pointcloud_asSegColored(plyDataPointsFlattened, outputFramePath_segColored)

            if DEBUG_FRAMEINDEX_LIMIT is not None and DEBUG_FRAMEINDEX_LIMIT <= frameIndex:
                break
            # -------------------------------------------

def setupGlobals():
    global ade20KToCarla
    import numpy as np
    import pandas as pd
    labelsMapping = pd.read_csv(ADE20K_TO_CARLA_MAPPING_CSV)
    # labelsMapping.head()
    ADE20K_labels = np.array(labelsMapping['Idx'])
    CARLA_labels = np.array(labelsMapping['CARLA_ID'])
    ade20KToCarla = {ADE20K_labels[k]: CARLA_labels[k] for k in range(len(ADE20K_labels))}
    ade20KToCarla[0] = 0

if __name__ == "__main__":
    setupGlobals()
    doPointCloudReconstruction(FILENAME_SAMPLE)

