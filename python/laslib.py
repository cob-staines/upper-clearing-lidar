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
        p0 = p0.drop(columns=[drop_columns])
    elif type(drop_columns) == list:
        p0 = p0.drop(columns=drop_columns)
    elif type(drop_columns) != type(None):
        raise Exception('"drop_columns" instance of unexpected class: ' + str(type(drop_columns)))

    # save to file
    p0.to_hdf(hdf5_out, key='las_data', mode='w', format='table')


def las_xyz_load(las_path, drop_class=None, keep_class=None):
    """
    Loads xyz data from .las (point cloud) file "las_in" dropping classes in "drop_class"
    :param las_path: file path to .las file
    :param drop_class: None, integer, or list of integers declaring point class(es) to be dropped
    :param keep_class: None, integer, or list of integers declaring point class(es) to be dropped
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
    if drop_class is None:
        drop_class_filter = np.full(classification.shape, True)
    elif type(drop_class) == int:
        drop_class_filter = (classification != drop_class)
    elif type(drop_class) == list:
        drop_class_filter = ~np.any(list(cc == classification for cc in drop_class), axis=0)
    else:
        raise Exception('"drop_class" instance of unexpected class: ' + str(type(drop_class)))

    if keep_class is None:
        keep_class_filter = np.full(classification.shape, True)
    elif type(keep_class) == int:
        keep_class_filter = (classification == keep_class)
    elif type(keep_class) == list:
        keep_class_filter = np.any(list(cc == classification for cc in keep_class), axis=0)
    else:
        raise Exception('"keep_class" instance of unexpected class: ' + str(type(keep_class)))

    # unload classification
    classification = None

    # filter xyz with class_filter
    las_xyz = p0[drop_class_filter & keep_class_filter]

    return las_xyz


def las_xyz_to_hdf5(las_in, hdf5_out, drop_class=None, keep_class=None):
    """
    Saves xyz points from las_path as hdf5 file to hdf5_path, dropping points of class(s) drop_class
    :param las_in: file path to .las file
    :param hdf5_out: file path to output .hdf5 file
    :param drop_class: None, integer, or list of integers declaring point class(es) to be dropped
    :return: None
    """
    import pandas as pd

    # load xyz from las_in
    las_data = las_xyz_load(las_in, drop_class=drop_class, keep_class=keep_class)

    p0 = pd.DataFrame({
        'x': las_data[:, 0],
        'y': las_data[:, 1],
        'z': las_data[:, 2]
    })

    p0.to_hdf(hdf5_out, key='las_data', mode='w', format='table')


def hemigen(hdf5_path, hemimeta):
    """
    Generates synthetic hemispherical image from xyz point cloud in hdf5 format.
    :param hdf5_path: file path to hdf5 file containing point cloud
    :param hemimeta: hemispherical metadata object containing required and optional metadata
        required metadata:
            hemimeta.origin: 3-tuple or list of 3-tuples corresponding to xyz coordinates of hemisphere center
            hemimeta.file_name
    :return: hm -- hemimeta in a pd.DataFrame, each row corresponding to each image
    """
    print("-------- Running Hemigen --------")

    import pandas as pd
    import numpy as np
    import matplotlib
    import h5py
    matplotlib.use('Agg')
    # matplotlib.use('TkAgg')  # use for interactive plotting
    import matplotlib.pyplot as plt
    import time
    import os

    tot_time = time.time()

    # convert to list of 1 if only one entry
    if hemimeta.origin.shape.__len__() == 1:
        hemimeta.origin = np.array([hemimeta.origin])
    if type(hemimeta.file_name) == str:
        hemimeta.file_dir = [hemimeta.file_dir]

    # QC: ensure origins and file_names have same length
    if hemimeta.origin.shape[0] != hemimeta.file_name.__len__():
        raise Exception('origin_coords and img_out_path have different lengths, execution halted.')

    # load data
    p0 = pd.read_hdf(hdf5_path, key='las_data', columns=["x", "y", "z"])

    # pre-plot
    fig = plt.figure(figsize=(hemimeta.img_size, hemimeta.img_size), dpi=hemimeta.img_resolution, frameon=False)
    # ax = plt.axes([0., 0., 1., 1.], projection="polar", polar=False)
    ax = plt.axes([0., 0., 1., 1.], projection="polar")
    # sp1 = ax.scatter(data.phi, data.theta, s=data.area, c="black")
    sp1 = ax.scatter([], [], s=[], c="black")
    ax.set_rmax(np.pi / 2)
    # ax.set_rticks([])
    ax.grid(False)
    ax.set_axis_off()
    fig.add_axes(ax)

    hm = pd.DataFrame({"id": hemimeta.id,
                       "file_name": hemimeta.file_name,
                       "file_dir": hemimeta.file_dir,
                       "x_utm11n": hemimeta.origin[:, 0],
                       "y_utm11n": hemimeta.origin[:, 1],
                       "elevation_m": hemimeta.origin[:, 2],
                       "src_las_file": hemimeta.src_las_file,
                       "las_class": hemimeta.src_keep_class,
                       "poisson_radius_m": hemimeta.poisson_sampling_radius,
                       "optimization_scalar": hemimeta.optimization_scalar,
                       "point_size_scalar": hemimeta.point_size_scalar,
                       "max_distance_m": hemimeta.max_distance,
                       "img_size_in": hemimeta.img_size,
                       "img_res_dpi": hemimeta.img_resolution,
                       "created_datetime": None,
                       "point_count": None,
                       "computation_time_s": None})

    # preallocate log file
    log_path = hemimeta.file_dir + "hemimetalog.csv"
    if not os.path.exists(log_path):
        with open(log_path, mode='w', encoding='utf-8') as log:
            log.write(", ".join(hm.columns) + '\n')
        log.close()

    for ii in range(0, hemimeta.origin.shape[0]):
        start = time.time()
        print("Generating " + hemimeta.file_name[ii] + " ...")

        p1 = p0.values - hemimeta.origin[ii]

        # if no max_radius, set to +inf
        if hemimeta.max_distance is None:
            hemimeta.max_distance = float("inf")

        # calculate r
        r = np.sqrt(np.sum(p1 ** 2, axis=1))
        # subset to within max_radius
        subset_f = r < hemimeta.max_distance
        r = r[subset_f]
        p1 = p1[subset_f]

        # flip over x axis for upward-looking perspective
        p1[:, 0] = -p1[:, 0]

        # calculate plot vars
        data = pd.DataFrame({'theta': np.arccos(p1[:, 2] / r),
                             'phi': np.arctan2(p1[:, 1], p1[:, 0]),
                             'area': ((1 / r) ** 2) * hemimeta.point_size_scalar})

        # plot
        sp1.set_offsets(np.c_[np.flip(data.phi), np.flip(data.theta)])
        sp1.set_sizes(data.area)

        # save figure to file
        fig.savefig(hemimeta.file_dir + hemimeta.file_name[ii])

        # log meta
        hm.loc[ii, "created_datetime"] = time.strftime('%Y-%m-%d %H:%M:%S')
        hm.loc[ii, "point_count"] = data.shape[0]
        hm.loc[ii, "computation_time_s"] = int(time.time() - start)

        # write to log file
        hm.iloc[ii:ii + 1].to_csv(log_path, encoding='utf-8', mode='a', header=False, index=False)

        print(str(ii + 1) + " of " + str(hemimeta.origin.shape[0]) + " complete: " + str(hm.computation_time_s[ii]) + " seconds")

    print("-------- Hemigen completed--------")
    print(str(hemimeta.origin.shape[0]) + " images generated in " + str(int(time.time() - tot_time)) + " seconds")

    return hm


class HemiMetaObj(object):
    def __init__(self):
        # preload metadata
        self.id = None
        self.file_name = None
        self.file_dir = None
        self.origin = None
        self.src_las_file = None
        self.src_keep_class = None
        self.optimization_scalar = None
        self.poisson_sampling_radius = None
        self.max_distance = None
        self.img_size = None
        self.img_resolution = None
        self.point_size_scalar = None


def las_traj(hdf5_path, traj_in):
    """

    :param hdf5_path: path to existing hdf5 file created using las_to hdf5
    :param traj_in: path to trajectory file corresponding to original las file
    :return:
    """

    import numpy as np
    import pandas as pd

    # load data from hdf5 file
    point_data = pd.read_hdf(hdf5_path, key='las_data', columns=['gps_time', 'x', 'y', 'z'])
    # add las key (True)
    las_data = point_data.assign(las=True)

    # load trajectory from csv
    traj = pd.read_csv(traj_in)
    # rename columns for consistency
    traj = traj.rename(columns={'Time[s]': "gps_time",
                                'Easting[m]': "easting_m",
                                'Northing[m]': "northing_m",
                                'Height[m]': "height_m"})
    # drop pitch, roll, yaw
    traj = traj[['gps_time', 'easting_m', 'northing_m', 'height_m']]
    # add las key (False)
    traj = traj.assign(las=False)

    # append traj to las, keeping track of las index
    outer = las_data[['gps_time', 'las']].append(traj, sort=False)
    outer = outer.reset_index()
    outer = outer.rename(columns={"index": "index_las"})

    # order by gps time
    outer = outer.sort_values(by="gps_time")

    # QC: check first and last entries are traj
    if (outer.las.iloc[0] | outer.las.iloc[-1]):
        raise Exception('LAS data exists outside trajectory time frame -- Suspect LAS/trajectory file mismatch')

    # set index as gps_time
    outer = outer.set_index('gps_time')

    # forward fill nan values
    interpolated = outer.fillna(method='ffill')

    # drop traj entries
    interpolated = interpolated[interpolated['las']]
    # reset to las index
    interpolated = interpolated.set_index("index_las")
    # drop las key column
    interpolated = interpolated[['easting_m', 'northing_m', 'height_m']]

    # concatenate with las_data horizontally by index
    merged = pd.concat([point_data, interpolated], axis=1, ignore_index=False)

    # distance from sensor
    p1 = np.array([merged.easting_m, merged.northing_m, merged.height_m])
    p2 = np.array([merged.x, merged.y, merged.z])
    squared_dist = np.sum((p1 - p2) ** 2, axis=0)
    merged = merged.assign(distance_from_sensor_m=np.sqrt(squared_dist))

    # angle from nadir
    dp = p1 - p2
    phi = np.arctan(np.sqrt(dp[0] ** 2 + dp[1] ** 2) / dp[2]) * 180 / np.pi  # in degrees
    merged = merged.assign(angle_from_nadir_deg=phi)

    # angle cw from north
    theta = np.arctan2(dp[0], (dp[1])) * 180 / np.pi
    merged = merged.assign(angle_cw_from_north_deg=theta)

    # select columns for output
    output = merged[["gps_time", "distance_from_sensor_m", "angle_from_nadir_deg", "angle_cw_from_north_deg"]]

    # save to hdf5 file
    output.to_hdf(hdf5_path, key='las_traj', mode='r+', format='table')


def las_poisson_sample(las_in, poisson_radius, classification=None, las_out=None):
    import pdal
    import time
    print("-------- Poisson Sampling --------")
    ps_start = time.time()

    # format file paths
    las_in = las_in.replace('\\', '/')
    if las_out is not None:
        las_out = las_out.replace('\\', '/')

    # do we filter?
    class_filter_bool = classification is not None
    if class_filter_bool & ~isinstance(classification, int):
        raise Exception('Only single classes are accepted in class filter currently.'
                        'Program additional handling if filtering with multiple classes is needed.')

    # do we save to file?
    save_bool = las_out is not None

    # json snippets
    json_open = """
            [
                "{inFile}","""
    json_class_filter = """
                {{
                "type":"filters.range",
                "limits":"Classification[{point_class}:{point_class}]"
                }},"""
    json_poisson = """
                {{
                    "type": "filters.sample",
                    "radius": "{radius}"
                }}"""
    json_save = """
                {{
                    "type": "writers.las",
                    "filename": "{outFile}"
                }}"""
    json_close = """
            ]
    """
    # compile json snippets
    json = json_open
    if class_filter_bool:
        json = json + json_class_filter
    json = json + json_poisson
    if save_bool:
        json = json + "," + json_save
    json = json + json_close

    json = json.format(inFile=las_in, outFile=las_out, radius=poisson_radius, point_class=classification)

    # execute json
    pipeline = pdal.Pipeline(json)
    count = pipeline.execute()
    arrays = pipeline.arrays

    ps_time = time.time() - ps_start
    print("Poisson sampling at radius of " + str(poisson_radius) + " meters completed in " +
          str(int(ps_time)) + " seconds")

    # return results as array
    return arrays