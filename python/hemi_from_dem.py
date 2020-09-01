import pandas as pd
import numpy as np
import laslib
import rastools

# build point list
dem_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\DEM\\19_149_dem_res_1.00m.bil'
# load dem
dem = rastools.raster_load(dem_in)
# list points where dem data exists
pts_index = np.where(dem.data != dem.no_data)
# convert to utm
pts_utm = dem.T1 * pts_index
# add all to df
pts = pd.DataFrame({'x_utm11n': pts_utm[0],
                    'y_utm11n': pts_utm[1],
                    'z_m': dem.data[pts_index],
                    'x_index': pts_index[1],
                    'y_index': pts_index[1]})
# add point id
pts = pts.reset_index()
pts.columns = ['id', 'x_utm11n', 'y_utm11n', 'z_m', 'x_index', 'y_index']

# add flag for UF
uf_plot = dem
uf_plot.data = np.full((dem.rows, dem.cols), 0)
uf_plot_dir = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\site_library\\uf_1m.tiff'
rastools.raster_save(uf_plot, uf_plot_dir, data_format='byte')
site_poly = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\site_library\\upper_forest_poly_UTM11N.shp'
rastools.raster_burn(uf_plot_dir, site_poly, 1)
uf_plot = rastools.raster_load(uf_plot_dir)

pts = pts.assign(uf=uf_plot.data[pts_index].astype(bool))
pts_dir = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\site_library\\1m_dem_points.csv'
pts.to_csv(pts_dir, index=False)

# build hemispheres
pts = pd.read_csv(pts_dir)
# filter to upper forest
pts = pts[pts.uf]

# hemi run
# define metadata object
hemimeta = laslib.HemiMetaObj()

# source las file
las_day = 19_149
hemimeta.src_las_file = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\LAS\\19_149_snow_off_classified_merged.las"
hemimeta.src_keep_class = 5
hemimeta.poisson_sampling_radius = 0.15  # meters (for no poisson sampling, specify 0)

# output file dir
hemimeta.file_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\opt\\poisson\\"

# max distance of points considered in image
hemimeta.max_distance = 50  # meters
hemi_m_above_ground = 2  # meters

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
        # laslib.las_poisson_sample(hemimeta.src_las_file, hemimeta.poisson_sampling_radius, classification=hemimeta.src_keep_class, las_out=las_poisson_path)  # takes 10 minutes
    else:
        raise Exception('hemimeta.poisson_sampling_radius should be a numeric >= 0 or None.')

# export las to hdf5
print("-------- Exporting to HDF5 --------")
hdf5_path = las_poisson_path.replace('.las', '.hdf5')
laslib.las_xyz_to_hdf5(las_poisson_path, hdf5_path, keep_class=hemimeta.src_keep_class)

hemimeta.id = pts.id
hemimeta.origin = np.array([pts.x_utmn11,
                            pts.y_utmn11,
                            pts.z_m + hemi_m_above_ground]).swapaxes(0, 1)


# point size
hemimeta.optimization_scalar = 10
footprint = 0.15  # in m
c = 2834.64  # meters to points
hemimeta.point_size_scalar = footprint**2 * c * hemimeta.optimization_scalar
hemimeta.file_name = ["las_" + str(las_day) + "_id_" + str(id) + "_pr_" + str(hemimeta.poisson_sampling_radius) +
                      "_os_" + str(hemimeta.optimization_scalar) + ".png" for id in pts.id]
hm = laslib.hemigen(hdf5_path, hemimeta, initial_index=0)
