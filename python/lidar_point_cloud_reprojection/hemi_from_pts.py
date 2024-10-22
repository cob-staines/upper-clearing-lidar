def main():
    """
    Configuration file for generating synthetic hemispherical images by lidar point cloud reprojection (eg. Moeser et al. 2014)
    at a set of given coordinates
        batch_dir: directory to save outputs
        las_in: las (point cloud) file to be used for reprojection
        pts_in: csv of coordinates including
    :return:
    """
    import pandas as pd
    import numpy as np
    import libraries.laslib
    import os

    #
    batch_dir = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\batches\\uf_1m_pr0_os0.063_snow_off_dem_offset.25_set_0\\'
    las_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\LAS\\19_149_las_proc_classified_merged.las"
    pts_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\site_library\\hemi_grid_points\\mb_65_r.25m_snow_off_offset.25\\dem_r.25_point_ids_1m subset.csv"


    # batch_dir = 'C:\\Users\\jas600\\workzone\\data\\hemigen\\uf_1m_pr0_os0.53_snow_off_dem_offset.25_set1\\'
    # las_in = "C:\\Users\\jas600\\workzone\\data\\ray_sampling\\sources\\19_149\\19_149_las_proc_classified_merged.las"
    # pts_in = 'C:\\Users\\jas600\\workzone\\data\\hemi_grid_points\\mb_65_r.25m_snow_off_offset.25\\dem_r.25_point_ids_1m subset.csv'

    # create batch dir if does not exist
    if not os.path.exists(batch_dir):
        os.makedirs(batch_dir)

    # build hemispheres
    pts = pd.read_csv(pts_in)

    # hemi run
    # define metadata object
    hemimeta = laslib.HemiMetaObj()

    # source las file
    las_day = 19_149
    hemimeta.src_las_file = las_in
    hemimeta.src_keep_class = [1, 5]  # range of classes or single class ([1, 5] passes all classes within 1-5)

    # Poisson sampling of point cloud, allows for thinning of high-density areas by limiting the minimum distance (radius) between accepted points.
    hemimeta.poisson_sampling_radius = 0  # meters (for no poisson sampling, specify 0).

    # output file dir
    hemimeta.file_dir = batch_dir + "outputs\\"
    if not os.path.exists(hemimeta.file_dir):
        os.makedirs(hemimeta.file_dir)

    # max distance of points considered in image
    hemimeta.max_distance = 50  # meters
    hemimeta.min_distance = 0  # meters
    hemimeta.max_phi = 65 * np.pi / 180  # radians
    hemi_m_above_ground = 0  # meters

    # image size
    hemimeta.img_size = 10  # in inches
    hemimeta.img_resolution = 100  # pixels/inch


    # poisson sample point cloud (src_las_in)
    if (hemimeta.poisson_sampling_radius is None) or (hemimeta.poisson_sampling_radius == 0):
        # skip poisson sampling
        las_poisson_path = hemimeta.src_las_file
        print("no Poisson sampling conducted")
    else:
        if hemimeta.poisson_sampling_radius > 0:
            # poisson sampling
            las_poisson_path = hemimeta.src_las_file.replace('.las', '_poisson_' + str(hemimeta.poisson_sampling_radius) + '.las')
            laslib.las_poisson_sample(hemimeta.src_las_file, hemimeta.poisson_sampling_radius, classification=hemimeta.src_keep_class, las_out=las_poisson_path)  # takes 10 minutes
        else:
            raise Exception('hemimeta.poisson_sampling_radius should be a numeric >= 0 or None.')

    # export las to hdf5
    print("-------- Exporting las to HDF5 --------")
    hdf5_path = las_poisson_path.replace('.las', '.hdf5')
    laslib.las_xyz_to_hdf5(las_poisson_path, hdf5_path, keep_class=hemimeta.src_keep_class)

    hemimeta.id = pts.id
    hemimeta.origin = np.array([pts.x_utm11n,
                                pts.y_utm11n,
                                pts.z_m + hemi_m_above_ground]).swapaxes(0, 1)

    # point size
    hemimeta.optimization_scalar = 0.063
    footprint = 0.15  # in m
    c = 2834.64  # meters to points
    hemimeta.point_size_scalar = footprint**2 * c * hemimeta.optimization_scalar
    hemimeta.file_name = ["las_" + str(las_day) + "_id_" + str(id) + "_pr_" + str(hemimeta.poisson_sampling_radius) +
                          "_os_" + str(hemimeta.optimization_scalar) + ".png" for id in pts.id]
    hm = laslib.hemigen(hdf5_path, hemimeta)

if __name__ == "__main__":
    main()