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
