drop_columns = None

def las_to_hdf5(las_in, hdf5_out, drop_columns=None):
    """
    Loads xyz data from .las (point cloud) file "las_in" dropping classes in "drop_class"
    :param las_in: file path to .las file
    :param hdf5_out: file path to output hdf5 file
    :param drop_columns: None, string, or list of strings declaring column names to be dropped before output to file
    :return: None
    """
    import laspy
    import pandas as pd

    # load las_in
    inFile = laspy.file.File(las_in, mode="r")
    # pull in xyz and classification
    p0 = pd.DataFrame({"gps_time": inFile.gps_time,
                       "x": inFile.x,
                       "y": inFile.y,
                       "z": inFile.z,
                       "classification": inFile.classification,
                       "intensity": inFile.intensity,
                       "num_returns": inFile.num_returns,
                       "return_num": inFile.return_num,
                       "scan_angle_rank": inFile.scan_angle_rank})
    # close las_in
    inFile.close()

    # create class_filter for drop_class in classification
    if type(drop_columns) == str:
        p0 = p0.drop(columns = [drop_columns])
    elif type(drop_columns) == list:
        p0 = p0.drop(columns = drop_columns)
    elif type(drop_columns) != type(None):
        raise Exception('"drop_columns" instance of unexpected class: ' + str(type(drop_columns)))

    # save to file
    p0.to_hdf(hdf5_out, key='data', mode='w', format='table')

def las_xyz_load(las_path, drop_class=None):
    """
    Loads xyz data from .las (point cloud) file "las_in" dropping classes in "drop_class"
    :param las_path: file path to .las file
    :param drop_class: None, integer, or list of integers declaring point class(es) to be dropped
    :return: las_xyz: numpy array of x, y, and z coordinates for
    """
    import laspy
    import numpy as np

    # load las_in
    inFile = laspy.file.File(las_path, mode="r")
    # pull in xyz and classification
    p0 = np.array([inFile.x,
                   inFile.y,
                   inFile.z]).transpose()
    classification = np.array(inFile.classification)
    # close las_in
    inFile.close()

    # create class_filter for drop_class in classification
    if drop_class == None:
        class_filter = np.full_like(classification, True)
    elif type(drop_class) == int:
        class_filter = (classification != drop_class)
    elif type(drop_class) == list:
        class_filter = ~np.any(list(cc == classification for cc in drop_class), axis=0)
    else:
        raise Exception('"drop_class" instance of unexpected class: ' + str(type(drop_class)))

    # unload classification
    classification = None

    # filter xyz with class_filter
    las_xyz = p0[class_filter]

    return las_xyz

def las_xyz_to_hdf5(las_path, hdf5_path, drop_class=None):
    """
    Saves xyz points from las_path as hdf5 file to hdf5_path, dropping points of class(s) drop_class
    :param las_path: file path to .las file
    :param hdf5_path: file path to output .hdf5 file
    :param drop_class: None, integer, or list of integers declaring point class(es) to be dropped
    :return: None
    """
    import h5py

    # load xyz from las_in
    las_data = las_xyz_load(las_path, drop_class=drop_class)

    h5f = h5py.File(hdf5_path, 'w')
    h5f.create_dataset('dataset', data=las_data)
    h5f.close()

def hemigen(hdf5_xyz_path, origin_coords, img_out_path, max_radius=None, point_size_scalar=1, img_size=10, img_res=100):
    """
    Generates synthetic hemispherical image from xyz point cloud in hdf5 format.
    :param hdf5_xyz_path: file path to hdf5 file containing point xyz coordinates
    :param origin_coords: numpy.array([x0, y0, z0]) of coordinates where hemispherical image will be centered
    :param img_out_path: file path of output image (should include file extension of desired image format)
    :param max_radius: distance from origin (numeric, in units of xyz coordinates) beyond which points will be dropped
    :param point_size_scalar: numeric for scaling the apparent size of points in hemispherical images
    :param img_size: dimension (in inches) of height and width of output image
    :param img_res: resolution (in pixels per inch) of output image
    :return: None
    """
    import pandas as pd
    import numpy as np
    import h5py
    import matplotlib
    matplotlib.use('Agg')
    # matplotlib.use('TkAgg')  # use for interactive plotting
    import matplotlib.pyplot as plt

    # open hdf5
    h5f = h5py.File(hdf5_xyz_path, 'r')
    # load xyz data
    las_xyz = h5f['dataset']

    # move to new origin
    p1 = las_xyz - origin_coords

    # close file
    h5f.close()

    # if no max_radius, set to +inf
    if max_radius == None:
        max_radius = float("inf")

    # calculate r
    r = np.sqrt(np.sum(p1 ** 2, axis=1))
    # subset to within max_radius
    subset_f = r < max_radius
    r = r[subset_f]
    p1 = p1[subset_f]

    # flip over x axis for upward-looking perspective
    p1[:, 0] = -p1[:, 0]

    # calculate plot vars
    data = pd.DataFrame({'theta': np.arccos(p1[:, 2] / r),
                         'phi': np.arctan2(p1[:, 1], p1[:, 0]),
                         'area': ((1 / r) ** 2) * point_size_scalar})

    # drop arrays for mem management
    p1 = None
    r = None
    subset_f = None

    # plot
    fig = plt.figure(figsize=(img_size, img_size), dpi=img_res, frameon=False)
    ax = plt.axes([0., 0., 1., 1.], projection="polar", polar=True)
    sp1 = ax.scatter(data.phi, data.theta, s=data.area, c="black")
    ax.set_rmax(np.pi / 2)
    ax.set_rticks([])
    ax.grid(False)
    ax.set_axis_off()
    fig.add_axes(ax)

    fig.savefig(img_out_path)
    print("done with " + img_out_path)

    data = None


def las_traj(las_in, traj_in):
    # las_traj takes in an las file "las_in" and a corresponding trajectory file "traj_in". The function then:
    #   -> merges files on gps_time
    #   -> interpolates trajectory to las_points
    #   -> calculates angle_from_nadir
    #   -> calculates distance_to_target
    #   -> returns laspy object

    # dependencies
    import laspy
    import pandas as pd
    import numpy as np

    # import las_in
    inFile = laspy.file.File(las_in, mode="r")
    # select dimensions
    las_data = pd.DataFrame({'gps_time': inFile.gps_time,
                             'x': inFile.x,
                             'y': inFile.y,
                             'z': inFile.z,
                             'intensity': inFile.intensity})
    inFile.close()
    las_data = las_data.assign(las=True)

    # import trajectory
    traj = pd.read_csv(traj_in)
    # rename columns for consistency
    traj = traj.rename(columns={'Time[s]': "gps_time",
                                'Easting[m]': "easting_m",
                                'Northing[m]': "northing_m",
                                'Height[m]': "height_m"})
    # throw our pitch, roll, yaw (at least until needed later...)
    traj = traj[['gps_time', 'easting_m', 'northing_m', 'height_m']]
    traj = traj.assign(las=False)

    # resample traj to las gps times and interpolate
    # outer merge las and traj on gps_time

    # proper merge takes too long, instead keep track of index
    outer = las_data[['gps_time', 'las']].append(traj, sort=False)
    outer = outer.reset_index()
    outer = outer.rename(columns={"index": "index_las"})

    # order by gps time
    outer = outer.sort_values(by="gps_time")
    # set index as gps_time for nearest neighbor interpolation
    outer = outer.set_index('gps_time')
    # interpolate by nearest neighbor

    interpolated = outer.interpolate(method="nearest")

    # resent index for clarity

    interpolated = interpolated[interpolated['las']]
    interpolated = interpolated.sort_values(by="index_las")
    interpolated = interpolated.reset_index()
    interpolated = interpolated[['easting_m', 'northing_m', 'height_m']]

    merged = pd.concat([las_data, interpolated], axis=1)
    merged = merged.drop('las', axis=1)

    # calculate point distance from track
    p1 = np.array([merged.easting_m, merged.northing_m, merged.height_m])
    p2 = np.array([merged.x, merged.y, merged.z])
    squared_dist = np.sum((p1 - p2) ** 2, axis=0)
    merged = merged.assign(distance_to_track=np.sqrt(squared_dist))

    # calculate angle from nadir
    dp = p1 - p2
    phi = np.arctan(np.sqrt(dp[0] ** 2 + dp[1] ** 2) / dp[2])*180/np.pi #in degrees
    merged = merged.assign(angle_from_nadir_deg=phi)

    return merged
