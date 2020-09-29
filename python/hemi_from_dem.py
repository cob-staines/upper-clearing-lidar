import pandas as pd
import numpy as np
import laslib
import rastools
import os

# build point list from DEM
dem_in = 'C:\\Users\\jas600\\workzone\\data\\hemigen\\hemi_lookups\\19_149_dem_r1.00m_q0.25_interpolated_min1.tif'
las_in = "C:\\Users\\jas600\\workzone\\data\\hemigen\\hemi_lookups\\19_149_las_proc_classified_merged.las"
site_poly = 'C:\\Users\\jas600\\workzone\\data\\hemigen\\hemi_lookups\\upper_forest_poly_UTM11N.shp'
batch_dir = 'C:\\Users\\jas600\\workzone\\data\\hemigen\\uf_1m_pr_0_os_0.5\\'

# create batch dir if does not exist
if not os.path.exists(batch_dir):
    os.makedirs(batch_dir)

# load dem into pd
pts = rastools.raster_to_pd(dem_in, 'z_m')
# add point id
pts = pts.reset_index()
pts.columns = ['id', 'x_utm11n', 'y_utm11n', 'x_index', 'y_index', 'z_m']

# # add flag for site (UF)
# load dem as template
uf_plot = rastools.raster_load(dem_in)
# fill data with zeros
uf_plot.data = np.full((uf_plot.rows, uf_plot.cols), 0)
# save to file
uf_plot_dir = batch_dir + 'uf_plot_over_dem.tiff'
rastools.raster_save(uf_plot, uf_plot_dir, data_format='byte')
# burn site polygon into plot data as ones
rastools.raster_burn(uf_plot_dir, site_poly, 1)
# load plot data
uf_plot = rastools.raster_load(uf_plot_dir)

# merge plot data with points
pts_index = (pts.x_index.values, pts.y_index.values)
pts = pts.assign(uf=uf_plot.data[pts_index].astype(bool))

# export point lookup as csv
pts_dir = batch_dir + '1m_dem_points.csv'
pts.to_csv(pts_dir, index=False)

# format point ids as raster
id_raster = rastools.raster_load(dem_in)
id_raster.data = np.full([id_raster.rows, id_raster.cols], id_raster.no_data).astype(int)
id_raster.data[pts_index] = pts.id
# save id raster to file
id_raster_out = batch_dir + '1m_dem_point_ids.tif'
rastools.raster_save(id_raster, id_raster_out, data_format="int32")

# build hemispheres
pts = pd.read_csv(pts_dir)
# filter to upper forest
pts = pts[pts.uf]

# hemi run
# define metadata object
hemimeta = laslib.HemiMetaObj()

# source las file
las_day = 19_149
hemimeta.src_las_file = las_in
hemimeta.src_keep_class = 5
hemimeta.poisson_sampling_radius = 0  # meters (for no poisson sampling, specify 0)

# output file dir
hemimeta.file_dir = batch_dir + "outputs\\"
if not os.path.exists(hemimeta.file_dir):
    os.makedirs(hemimeta.file_dir)

# max distance of points considered in image
hemimeta.max_distance = 50  # meters
hemimeta.min_distance = .5  # meters
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
print("-------- Exporting to HDF5 --------")
hdf5_path = las_poisson_path.replace('.las', '.hdf5')
laslib.las_xyz_to_hdf5(las_poisson_path, hdf5_path, keep_class=hemimeta.src_keep_class)

hemimeta.id = pts.id
hemimeta.origin = np.array([pts.x_utm11n,
                            pts.y_utm11n,
                            pts.z_m + hemi_m_above_ground]).swapaxes(0, 1)


# point size
hemimeta.optimization_scalar = 0.5
footprint = 0.15  # in m
c = 2834.64  # meters to points
hemimeta.point_size_scalar = footprint**2 * c * hemimeta.optimization_scalar
hemimeta.file_name = ["las_" + str(las_day) + "_id_" + str(id) + "_pr_" + str(hemimeta.poisson_sampling_radius) +
                      "_os_" + str(hemimeta.optimization_scalar) + ".png" for id in pts.id]
hm = laslib.hemigen(hdf5_path, hemimeta, initial_index=0)
