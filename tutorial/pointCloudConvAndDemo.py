# This script is used to convert from our ply representation to Open3D compatible one
# Plus check locally how Open3D would perform
import os
import numpy as np
import time
import pipeline_commons
import ReconstructionUtils

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
                  'property uchar red',
                  'property uchar green',
                  'property uchar blue',
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

def do_conversion_toO3D(srcPath, targetPath):
    from plyfile import PlyData, PlyElement
    plydata = PlyData.read(srcPath)
    nr_points = plydata.elements[0].count
    pointcloud3D = np.array([plydata['vertex'][k] for k in range(nr_points)])
    newpointcloud3d = []
    for point in pointcloud3D:
        x,y,z,r,g,b, label = point

        """
        r /= 255.0
        g /= 255.0
        b /= 255.0
        """

        newpointcloud3d.append((x, y, z, r, g, b, label))

    save_3d_pointcloud(newpointcloud3d, targetPath)


#do_conversion_toO3D(Waymo_X_orig, Waymo_X)

def do_conv_folder(basePath, START_FRAME, END_FRAME, doSegToo = False):
    for i in range(START_FRAME, END_FRAME+1):
        print("## convering frame ", i)
        filename_in = os.path.join(basePath, "combined_carla_moving_f{0:05d}.ply".format(i))
        filename_out = os.path.join(basePath, "combined_carla_moving_f{0:05d}_conv.ply".format(i))

        print("RGB..")
        if not os.path.exists(filename_out):
            do_conversion_toO3D(filename_in, filename_out)

        print("seg..")
        if doSegToo:
            filename_in = os.path.join(basePath, "combined_carla_moving_segColor_f{0:05d}.ply".format(i))
            filename_out = os.path.join(basePath, "combined_carla_moving_segColor_f{0:05d}_conv.ply".format(i))

            if not os.path.exists(filename_out):
                do_conversion_toO3D(filename_in, filename_out)


##################################################################################


if False:
    sceneBasePath = "/home/ciprian/pres/WAYMOOUTPUT/15646511153936256674_1620_000_1640_000" #"C:/Users/Ciprian/OneDrive - University of Bucharest, Faculty of Mathematics and Computer Science/IMAR_Work/New folder/Scene18311"
    do_conv_folder(sceneBasePath, 0, 197, doSegToo=True)

if True:
    # basePath = "C:/Users/Ciprian/OneDrive - University of Bucharest, Faculty of Mathematics and Computer Science/IMAR_Work/New folder"
    basePath = "/home/ciprian/pres/WAYMOOUTPUT/18311996733670569136_5880_000_5900_000"

    DEMO_LOW_QUALITY = True
    # A low quality voxelized example...
    if DEMO_LOW_QUALITY == True:
        Waymo_X = os.path.join(basePath, "combined_carla_moving_no_res_conv.ply") #
        Waymo_X_noiseCut = os.path.join(basePath, "combined_carla_moving_no_res_simple.ply") #os.path.join(basePath, "combined_carla_moving - Copy_conv_2.ply")
        scale = 5.0
    elif DEMO_LOW_QUALITY == False:
        ### A High quality float example..
        Waymo_X = os.path.join(basePath, "combined_carla_moving - Copy_conv.ply")
        Waymo_X_noiseCut = os.path.join(basePath, "combined_carla_moving - Copy_conv_2.ply")
        scale = 1.0

    NEEDS_CONVERSION = False
    NEEDS_REMOVE_OUTLIER = False

    if NEEDS_CONVERSION: # Do conversion ?
        Waymo_X_orig = os.path.join(basePath, "combined_carla_moving_no_res.ply")
        start_time = time.time()
        do_conversion_toO3D(Waymo_X_orig, Waymo_X)
        print("--- %s seconds ---" % (time.time() - start_time))
        exit(0)

    import open3d as o3d
    print("Load a ply point cloud, print it, and render it")


    if NEEDS_REMOVE_OUTLIER:
        pcd = o3d.io.read_point_cloud(Waymo_X)
        selectedIndices, excludedIndices = ReconstructionUtils._noiseRemovalWithStatsAndKnn(pcd, scale=scale, debug=True)

        # Save the thing on disk
        pcd = pcd.select_by_index(excludedIndices, invert=True)
        o3d.io.write_point_cloud(Waymo_X_noiseCut, pcd)
        # exit(0)

    pcd = o3d.io.read_point_cloud(Waymo_X_noiseCut)#)Waymo_X_noiseCut)

    #print(pcd)
    #print(np.asarray(pcd.points))
    #print(np.asarray(pcd.colors))
    #print(pcd.has_colors())

    o3d.visualization.draw_geometries([pcd])

