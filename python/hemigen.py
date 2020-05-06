def las_load(las_in):
    import laspy
    import numpy as np

    # import las file "las_in"
    inFile = laspy.file.File(las_in, mode="r")
    ####
    p0 = np.array([inFile.x,
                   inFile.y,
                   inFile.z]).transpose()
    classification = np.array(inFile.classification)

    inFile.close()

    class_filter = (classification != 7) & (classification != 8)
    classification = None

    # remove noise
    las_xyz = p0[class_filter]

    return las_xyz

def las_to_hdf5(las_path, hdf5_path):
    import h5py
    las_data = las_load(las_path)

    h5f = h5py.File(hdf5_path, 'w')
    h5f.create_dataset('dataset', data=las_data)
    h5f.close()

def hemigen(las_hdf5, origin, fig_out, max_radius, point_size_scalar, fig_size, fig_dpi):
    import pandas as pd
    import numpy as np
    import h5py
    import matplotlib
    matplotlib.use('Agg')
    # matplotlib.use('TkAgg')  # use for interactive plotting
    import matplotlib.pyplot as plt

    # load hdf5
    hdf5_file = h5py.File(las_hdf5, 'r')
    las_xyz = hdf5_file['dataset']

    # move to new origin
    p1 = las_xyz - origin
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

    # mem management
    p1 = None
    r = None
    subset_f = None

    # plot
    fig = plt.figure(figsize=(fig_size, fig_size), dpi=fig_dpi, frameon=False)
    ax = plt.axes([0., 0., 1., 1.], projection="polar", polar=True)
    sp1 = ax.scatter(data.phi, data.theta, s=data.area, c="black")
    ax.set_rmax(np.pi / 2)
    ax.set_rticks([])
    ax.grid(False)
    ax.set_axis_off()
    fig.add_axes(ax)

    fig.savefig(fig_out)
    print("done with " + fig_out)

    data = None
