import pandas as pd
import numpy as np
import laslib
import rastools
import os

# build point list from DEM
dem_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\DEM\\19_149_dem_res_1.00m.bil'
uf_poly = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\site_library\\upper_forest_poly_UTM11N.shp'
uls_poly = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\site_library\\dens_site_poly_clipped.shp'
batch_dir = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\hemi_grid_points\\'

# dem_in = 'C:\\Users\\jas600\\workzone\\data\\hemigen\\hemi_lookups\\19_149_dem_r1.00m_q0.25_interpolated_min1.tif'
# site_poly = 'C:\\Users\\jas600\\workzone\\data\\hemigen\\hemi_lookups\\upper_forest_poly_UTM11N.shp'
# batch_dir = 'C:\\Users\\jas600\\workzone\\data\\hemigen\\uf_1m_pr_0_os_0.5\\'

# create batch dir if does not exist
if not os.path.exists(batch_dir):
    os.makedirs(batch_dir)

# load dem into pd
pts = rastools.raster_to_pd(dem_in, 'z_m')
# add point id
pts = pts.reset_index()
pts.columns = ['id', 'x_utm11n', 'y_utm11n', 'x_index', 'y_index', 'z_m']

# # add flag for (UF)
# load dem as template
site_plot = rastools.raster_load(dem_in)
# fill data with zeros
site_plot.data = np.full((site_plot.rows, site_plot.cols), 0)
# save to file
uf_plot_dir = batch_dir + 'uf_plot_over_dem.tiff'
rastools.raster_save(site_plot, uf_plot_dir, data_format='byte')
# burn site polygon into plot data as ones
rastools.raster_burn(uf_plot_dir, uf_poly, 1)
# load plot data
uf_plot = rastools.raster_load(uf_plot_dir)

# # add flag for upper lidar site (ULS)
# load dem as template
site_plot = rastools.raster_load(dem_in)
# fill data with zeros
site_plot.data = np.full((site_plot.rows, site_plot.cols), 0)
# save to file
uls_plot_dir = batch_dir + 'uls_plot_over_dem.tiff'
rastools.raster_save(site_plot, uls_plot_dir, data_format='byte')
# burn site polygon into plot data as ones
rastools.raster_burn(uls_plot_dir, uls_poly, 1)
# load plot data
uls_plot = rastools.raster_load(uls_plot_dir)

# merge plot data with points
pts_index = (pts.x_index.values, pts.y_index.values)
pts = pts.assign(uf=uf_plot.data[pts_index].astype(bool), uls=uls_plot.data[pts_index].astype(bool))

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

# point subsets

# 3-m grid over uls
# load 3-m template
temp_3m_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\raster_templates\\dummy_template_3m.tif"
pts_3m = rastools.raster_to_pd(temp_3m_in, 'dummy_3m')

# merge with pts along x_coors, y_coords
merged_3m = pd.merge(pts, pts_3m, left_on=['x_utm11n', 'y_utm11n'], right_on=['x_coord', 'y_coord'], how='left', suffixes=('', '_3m'))

# filter to 3m points
subset_3m = merged_3m[~np.isnan(merged_3m.x_coord)]
subset_3m = subset_3m[['id', 'x_utm11n', 'y_utm11n', 'x_index', 'y_index', 'z_m', 'uf', 'uls', 'x_index_3m', 'y_index_3m']]

# filter to uls
uls_3m = subset_3m[subset_3m.uls]
# export point lookup as csv
pts_dir = batch_dir + '1m_dem_points_3m_subgrid_uls.csv'
uls_3m.to_csv(pts_dir, index=False)