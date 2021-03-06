import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys


tf.enable_eager_execution()
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset


FILENAME = "/home/ciprian/Downloads/Waymo/segment-10023947602400723454_1120_000_1140_000_with_camera_labels.tfrecord" #'frames'
DEBUG_SEGMENTATION_ENABLED = True # If True, will dump input.output of images from segmentation
NO_CAMERA_INDEX = -1
NO_LABEL_POINT = 0

# Dummy function to do object detection using denset121 using a pretrained pytorch
def test_object_detection(image_bytes = None):
    import torch
    import io
    import torchvision.transforms as transforms
    from torchvision import models
    from PIL import Image

    # Make sure to pass `pretrained` as `True` to use the pretrained weights:
    model = models.densenet121(pretrained=True)
    # Since we are using our model only for inference, switch to `eval` mode:
    model.eval()

    def transform_image(image_bytes):
        my_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                [0.485, 0.456, 0.406],
                                                [0.229, 0.224, 0.225])])
        image = Image.open(io.BytesIO(image_bytes))
        return my_transforms(image).unsqueeze(0)

    def get_prediction(image_bytes):
        tensor = transform_image(image_bytes=image_bytes)
        outputs = model.forward(tensor)
        _, y_hat = outputs.max(1)
        return y_hat

    if image_bytes is None:
        with open("/home/ciprian/animals_hero_giraffe_1_0.jpg", 'rb') as f:
            image_bytes = f.read()

    print(get_prediction(image_bytes=image_bytes))


def show_camera_image(camera_image, camera_labels, layout, cmap=None):
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

def get_range_image(laser_name, return_index):
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

######################################################

# 1. Iterate over frame by frame of a segment
dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
for index, data in enumerate(dataset):
    # Read the frame in bytes
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))
    break

(range_images, camera_projections,
 range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(
    frame)
#print(frame.context)


# 2. Print some cameras images in the data
plt.figure(figsize=(25, 20))
for index, image in enumerate(frame.images):
    # test_object_detection(image.image)
    show_camera_image(image, frame.camera_labels, [3, 3, index+1])

# 3. Print range images
plt.figure(figsize=(64, 20))
frame.lasers.sort(key=lambda laser: laser.name)
show_range_image(get_range_image(open_dataset.LaserName.TOP, 0), 1)
show_range_image(get_range_image(open_dataset.LaserName.TOP, 1), 4)


# 4. Convert images to point cloud and view their content
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


returnsIndicesToUse = [0,1]
points_byreturn = [points_all_ri0, points_all_ri2]
cp_points_byreturn = [cp_points_all_ri0, cp_points_all_ri2]

'''
# First return
print(points_all.shape)
print(cp_points_all.shape)
print(points_all[0:2])
for i in range(5):
  print(points[i].shape)
  print(cp_points[i].shape)

# Second return
print(points_all_ri2.shape)
print(cp_points_all_ri2.shape)
print(points_all_ri2[0:2])
for i in range(5):
  print(points_ri2[i].shape)
  print(cp_points_ri2[i].shape)
'''
#from IPython.display import Image, display
#display(Image('3d_point_cloud.png'))

# 5. Project camera to image

############ SOme util MOVE THEM
from plyfile import PlyData, PlyElement

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

def read_3D_pointcloud(filename, file_end='/dense/fused_text.ply'):
    plydata_3Dmodel = PlyData.read(filename + file_end)
    nr_points = plydata_3Dmodel.elements[0].count
    pointcloud_3D = np.array([plydata_3Dmodel['vertex'][k] for k in range(nr_points)])
    return pointcloud_3D

#########################


# Demo: save all 3d points in a ply file - NO COLOR
isPlyDemoEnabled = True
isRangeColorShowingEnabled = False
if isPlyDemoEnabled:
    plyDataPoints = []
    for returnIndex in returnsIndicesToUse:
        points_data = points_byreturn[returnIndex]
        for point3D in points_data:
            plyDataPoints.append([*point3D, 255, 0, 0, 1])
    save_3d_pointcloud(plyDataPoints, "test.ply")

    plyDataPoints_Read = read_3D_pointcloud("test.ply", "")
    #exit(0)



###########################################
from scipy import stats

# Aggregates all points given in the output dictionary, for each 3D point cloud will add the R G B in the scene and the segmentation label
# unprojectedPoints is True if the points have no label or camera projection image
def processPoints(points3D_and_cp, outPlyDataPoints, imageCameraIndex = NO_CAMERA_INDEX):
    for point in points3D_and_cp:
        x,y,z = point[0:3]
        key = (x, y, z)

        camX, camY = None, None
        R, G, B = 0, 0, 0
        label = NO_LABEL_POINT

        if imageCameraIndex != NO_CAMERA_INDEX:
            camX, camY = point[3:5]

            # TODO:
            R, G, B = 255, 0, 0
            label = 1


        if key not in outPlyDataPoints:
            outPlyDataPoints[key] = []
        outPlyDataPoints[key].append((R, G, B, label))


# Takes the output dictionary build as above by ProcessPoints method and returns a flattened list
def convertDictPointsToList(inPlyDataPoints):
    outPlyDataPoints = []
    for keyPoint3D, pointData in inPlyDataPoints.items():
        x, y, z = keyPoint3D[0], keyPoint3D[1], keyPoint3D[2]

        # Select the mode label from the point list
        pointData = np.array(pointData).reshape(-1, 4) # R,G,B, seg label
        mode, count = stats.mode(pointData[:, 3])
        votedLabel = mode[0]

        #COMMENT THIS - Debug to see multiple points on the same coordinate:
        if pointData.shape[0] > 1:
            print("P: ({:.2f} {:.2f} {:.2f}) {}".format(x, y, z, pointData))

        # end debug

        votedRGB = None
        # Take first RGB corresponding to the mode
        for dataIndex in range(pointData.shape[0]):
            if pointData[dataIndex, 3] == votedLabel:
                votedRGB = (x, y, z, pointData[dataIndex, 0], pointData[dataIndex, 1], pointData[dataIndex, 2])
                break

        outPlyDataPoints.append((*votedRGB, votedLabel))
    return outPlyDataPoints

# Here we gather all ply data in format: {(x,y,z) : [r g b label]], for all points in the point cloud.
# Why dictionary ? Because we might want to discretize from original space to a lower dimensional space so same x,y,z from original data might go into the same chunk
plyDataPoints = {}


# For each return
for returnIndex in returnsIndicesToUse:
    image_indices_to_project = [0,1,2,3,4]

    # Put together [projected camera data, 3D points] on the same row, convert to tensor
    cp_points_all_concat = np.concatenate([cp_points_byreturn[returnIndex], points_byreturn[returnIndex]], axis=-1)
    cp_points_all_concat_tensor = tf.constant(cp_points_all_concat)

    # Compute the distance between lidar points and vehicle frame origin for each 3d point
    distances_all_tensor = tf.norm(points_byreturn[returnIndex], axis=-1, keepdims=True) if isRangeColorShowingEnabled is True else None

    # Create a tensor with all projected [projected camera data] and one with 3d points
    cp_points_all_tensor = tf.constant(cp_points_byreturn[returnIndex], dtype=tf.int32)
    points_3D_all_tensor = tf.constant(points_byreturn[returnIndex], dtype=tf.float32)

    # For each camera image index
    image_projections_indices = [0, 1]
    for imageProjIndex in image_projections_indices:
        for image_index in image_indices_to_project:
            images = getSortedImagesFromFrameData(frame)

            # A mask with True where the camera projection points where on this image index
            mask = tf.equal(cp_points_all_tensor[..., 0 if imageProjIndex == 0 else 3], images[image_index].name)

            # Now select only camera projection data and 3D points in vehicle frame associated with this camera index
            cp_points_all_tensor_camera = tf.cast(tf.gather_nd(cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)
            points_3D_all_tensor_camera = tf.cast(tf.gather_nd(points_3D_all_tensor, tf.where(mask)), dtype=tf.float32)

            # Select only the cp points associated with the iterated camera projection index
            cp_points_all_tensor_camera_byProjIndex = cp_points_all_tensor_camera[..., 1:3] if imageProjIndex == 0 else cp_points_all_tensor_camera[..., 4:6]

            # Associate on each row one 3D point with its corresponding image camera projection (this index being iterated on)
            # We have now on each row:  [(x,y,z, camX, camY)]
            points3D_and_cp = tf.concat([points_3D_all_tensor_camera, cp_points_all_tensor_camera_byProjIndex], axis=-1).numpy()

            # Gather these points in the output dictionary
            processPoints(points3D_and_cp, plyDataPoints, imageCameraIndex = image_index)

            # Demo showing...
            if isRangeColorShowingEnabled:
                distances_all_tensor_camera = tf.gather_nd(distances_all_tensor, tf.where(mask))

                # Associate on each row, with each x,y in camera space, the distance from vehicle frame to the 3D point lidar
                projected_points_all_from_raw_data = tf.concat(
                    [cp_points_all_tensor_camera[..., 1:3], distances_all_tensor_camera] if imageProjIndex == 0 else \
                    [cp_points_all_tensor_camera[..., 4:6], distances_all_tensor_camera],
                    axis=-1).numpy()

                plot_points_on_image(projected_points_all_from_raw_data,
                                     images[image_index], rgba, point_size=5.0)

    # Add all point cloud points which are not not labeled (not found in a projected camera image)
    mask = tf.equal(cp_points_all_tensor[..., 0], 0) # 0 is the index for unprojected camera point, on first cp index
    points_3D_all_unprojected_tensor = tf.cast(tf.gather_nd(points_3D_all_tensor, tf.where(mask)), dtype=tf.float32)
    processPoints(points_3D_all_unprojected_tensor.numpy(), plyDataPoints, imageCameraIndex = NO_CAMERA_INDEX)

plyDataPointsFlattened = convertDictPointsToList(plyDataPoints)
save_3d_pointcloud(plyDataPointsFlattened, "test2.ply") # TODO: save one file with test_x.ply, using RGB values, another one test_x_seg.ply using segmented data.

#plt.show()
