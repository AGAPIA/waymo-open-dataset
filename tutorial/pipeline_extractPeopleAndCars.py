import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import pipeline_commons
import pickle

#tf.debugging.set_log_device_placement(True)

tf.enable_eager_execution()
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

def get_min_max_3d_box_corners(boxes, vehicleToWorldTransform, worldToReferencePointTransform, useGlobalPoseTransform, name=None):
  """Given a set of upright boxes, return its 2 min and max corner points
     in world (global) space frame

  Args:
    boxes: tf Tensor [N, 7]. The inner dims are [center{x,y,z}, length, width,
      height, heading].
    vehicleToWorldTransform: None or a 4x4 transformation if needed from local to global space
    name: the name scope.

  Returns:
    corners: tf Tensor [N, 3, 2].
  """
  with tf.compat.v1.name_scope(name, 'GetMinMax3dBoxCorners', [boxes]):
    center_x, center_y, center_z, length, width, height, heading = tf.unstack(
        boxes, axis=-1)

    if useGlobalPoseTransform:
        vehicleToWorldRotation = vehicleToWorldTransform[0:3, 0:3]
        vehicleToWorldTranslation = vehicleToWorldTransform[0:3, 3]
        worldToReferenceRotation = worldToReferencePointTransform[0:3, 0:3]
        worldToReferenceTranslation = worldToReferencePointTransform[0:3, 3]

    # Step 1: get the box corners in local space
    # [N, 3, 3]
    rotation = transform_utils.get_yaw_rotation(heading)
    # [N, 3]
    translation = tf.stack([center_x, center_y, center_z], axis=-1)

    l2 = length * 0.5
    w2 = width * 0.5
    h2 = height * 0.5

    # [N, 8, 3]
    corners = tf.reshape(
        tf.stack([
            l2, w2, -h2, -l2, w2, -h2, -l2, -w2, -h2, l2, -w2, -h2, l2, w2, h2,
            -l2, w2, h2, -l2, -w2, h2, l2, -w2, h2
        ],
            axis=-1), [-1, 8, 3])
    # [N, 8, 3]
    # Step 2: transform the box corners from AABB to OOBB in local space
    corners = tf.einsum('nij,nkj->nki', rotation, corners) + tf.expand_dims(translation, axis=-2)

    if useGlobalPoseTransform:
        # [N, 8, 3]
        # Step 3: transform the box corners from OOBB in local space to OOBB in world space
        corners = tf.einsum('jk,nik->nij', vehicleToWorldRotation, corners) + tf.expand_dims(vehicleToWorldTranslation, axis=-2)

        # Step 4: transform from 00BB in world space to OOBB in reference frame (i.e. relative to the first reference vehicle position)
        corners = tf.einsum('jk,nik->nij', worldToReferenceRotation, corners) + tf.expand_dims(worldToReferenceTranslation, axis=-2)

    corners = corners.numpy()
    #print(corners)
    min_corners = corners.min(axis=1)
    max_corners = corners.max(axis=1)
    #print(min_corners)
    #print(max_corners)
    minMaxCorners = np.stack([min_corners, max_corners], axis=1).transpose(0, 2, 1)
    return minMaxCorners

def debugTest():
    # A simple test to prove correctness / debug the above operations
    import transformations as transf
    zaxis = [0, 0, 1]
    Rz = transf.rotation_matrix(np.pi/2, zaxis)
    Ts = transf.translation_matrix([10, 10, 0])
    transform = Ts.dot(Rz)
    bboxes = [[1.0,1.0,1.0, 10.0, 5.0, 3.0, 0.0], [2.0,2.0,2.0, 10.0, 5.0, 3.0, 0.0]]
    corners = get_min_max_3d_box_corners(bboxes, transform)
    print(corners)

def do_PeopleAndVehiclesExtraction(segmentPath, globalParams):
    # 1. Iterate over frame by frame of a segment
    dataset = tf.data.TFRecordDataset(segmentPath, compression_type='')
    lidarLabels = None

    pedestrians_data = {}
    vehicle_data = {}
    useGlobalPoseTransform = True
    worldToReferencePointTransform = None

    for frameIndex, data in enumerate(dataset):
        if (globalParams.FRAMEINDEX_MIN > frameIndex) and frameIndex != 0: # We need first frame reference point !!:
            continue
        if globalParams.FRAMEINDEX_MAX < frameIndex:
            break

        # Read the frame in bytes
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        lidarLabels = frame.laser_labels

        # All cars and pedestrians are relative to the car with sensors attached
        # We can transform to world positions using frame.pose.transform.
        # But to get coordinates back to the reference frame - WHICH IS THE INITIAL POSITION OF THE CAR - we use the below inverse matrix
        frame_thisPose = np.reshape(np.array(frame.pose.transform).astype(np.float32), [4, 4])
        if frameIndex == 0:
            worldToReferencePointTransform = np.linalg.inv(frame_thisPose)

        #print(f'There are {len(lidarLabels)} lidar labels')

        # Transform all boxes for this frame to the global world coordinates and retain minMax points from each
        allBoxes = [None] * len(lidarLabels)
        for entityIndex, label in enumerate(lidarLabels):
            box = label.box
            allBoxes[entityIndex] = [box.center_x, box.center_y, box.center_z, box.length, box.width, box.height, box.heading]


        allBoxesMinMax = get_min_max_3d_box_corners(allBoxes, vehicleToWorldTransform=frame_thisPose,
                                                    worldToReferencePointTransform=worldToReferencePointTransform,
                                                    useGlobalPoseTransform=useGlobalPoseTransform)
        assert allBoxesMinMax.shape[1] == 3 and allBoxesMinMax.shape[2] == 2, 'Incorrect format of data'

        # Write info for this frame
        if pedestrians_data.get(frameIndex) == None:
            pedestrians_data[frameIndex] = {}
        if vehicle_data.get(frameIndex) == None:
            vehicle_data[frameIndex] = {}

        for entityIndex, label in enumerate(lidarLabels):
            if label.type != label.TYPE_VEHICLE and label.type != label.TYPE_CYCLIST and label.type != label.TYPE_PEDESTRIAN:
                continue

            boxMinMax = allBoxesMinMax[entityIndex]

            isVehicle = label.type == label.TYPE_VEHICLE or label.type == label.TYPE_CYCLIST
            isPedestrian = label.type == label.TYPE_PEDESTRIAN
            targetOutputDict = vehicle_data[frameIndex] if isVehicle else pedestrians_data[frameIndex]

            assert targetOutputDict.get(label.id) is None, "This entity Id is already in the set !!"
            targetOutputDict[label.id] = { "BBMinMax" : boxMinMax }

        #break

    #print("Vehicles...\n", vehicle_data)
    #print("Pedestrians...\n", pedestrians_data)

    # Save people.p and cars.p
    segmentName = pipeline_commons.extractSegmentNameFromPath(segmentPath)
    segmentFiles_OutputPath = os.path.join(globalParams.MOTION_OUTPUT_BASEFILEPATH, segmentName)

    filepathsAndDictionaries = {'pedestrians' : (pedestrians_data, os.path.join(segmentFiles_OutputPath, "people.p")),
                                 'cars' : (vehicle_data, os.path.join(segmentFiles_OutputPath, "cars.p"))
                                }
    for key, value in filepathsAndDictionaries.items():
        dataObj = value[0]
        filePath = value[1]
        with open(filePath, mode="wb") as fileObj:
            pickle.dump(dataObj, fileObj, protocol=2) # Protocol 2 because seems to be compatible between 2.x and 3.x !

if __name__ == "__main__": # 'frames'
    import pipeline_params
    pipeline_params.globalParams.FRAMEINDEX_MIN = 0
    pipeline_params.globalParams.FRAMEINDEX_MAX = 5

    do_PeopleAndVehiclesExtraction(pipeline_params.FILENAME_SAMPLE[0], pipeline_params.globalParams)
