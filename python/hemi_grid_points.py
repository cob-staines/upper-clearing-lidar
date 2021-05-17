import pandas as pd
import numpy as np
import rastools
import os

batch_dir = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\site_library\\hemi_grid_points\\mb_65_r.25m_snow_on_offset0\\'

# build point list from DEM
dem_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_052\\19_052_las_proc\\OUTPUT_FILES\\DEM\\interpolated\\19_052_dem_interpolated_r.25m.tif'  # snow-on
# dem_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\DEM\\interpolated\\19_149_dem_interpolated_r.25m.tif'  # snow-off

vertical_offset = 0

mb_65_poly = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\site_library\\mb_65_poly.shp'
mb_15_poly = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\site_library\\mb_15_poly.shp'
uf_poly = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\site_library\\upper_forest_poly_UTM11N.shp'
uc_poly = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\site_library\\upper_clearing_poly_UTM11N.shp'

# for plot mappings
resolution = ['.05', '.10', '.25', '1.00']
template_scheme = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\TEMPLATES\\19_149_all_point_density_r<RES>m.bil'

# dem_in = 'C:\\Users\\jas600\\workzone\\data\\hemigen\\hemi_lookups\\19_149_dem_r1.00m_q0.25_interpolated_min1.tif'
# site_poly = 'C:\\Users\\jas600\\workzone\\data\\hemigen\\hemi_lookups\\upper_forest_poly_UTM11N.shp'
# batch_dir = 'C:\\Users\\jas600\\workzone\\data\\hemigen\\uf_1m_pr_0_os_0.5\\'

# create batch dir if does not exist
if not os.path.exists(batch_dir):
    os.makedirs(batch_dir)

pts = rastools.raster_to_pd(dem_in, 'z_m', include_nans=True)
pts.z_m = pts.z_m + vertical_offset  # shift z_m by vertical offset


# add point id
pts = pts.reset_index()
pts.columns = ['id', 'x_utm11n', 'y_utm11n', 'x_index', 'y_index', 'z_m']


# # add flag for mb_65
# load dem as template
site_plot = rastools.raster_load(dem_in)
# fill data with zeros
site_plot.data = np.full((site_plot.rows, site_plot.cols), 0)
# save to file
mb_65_plot_dir = batch_dir + 'mb_65_plot_over_dem.tiff'
rastools.raster_save(site_plot, mb_65_plot_dir, data_format='byte')
# burn site polygon into plot data as ones
rastools.raster_burn(mb_65_plot_dir, mb_65_poly, 1)
# load plot data
mb_65_plot = rastools.raster_load(mb_65_plot_dir)

# # add flag for mb_15
# load template
site_plot = rastools.raster_load(dem_in)
# fill data with zeros
site_plot.data = np.full((site_plot.rows, site_plot.cols), 0)
# save to file
mb_15_plot_dir = batch_dir + 'mb_15_plot_over_dem.tiff'
rastools.raster_save(site_plot, mb_15_plot_dir, data_format='byte')
# burn site polygon into plot data as ones
rastools.raster_burn(mb_15_plot_dir, mb_15_poly, 1)
# load plot data
mb_15_plot = rastools.raster_load(mb_15_plot_dir)


# # add flag for (UF)
# load template
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



# merge plot data with points
pts_index = (pts.y_index.values, pts.x_index.values)
pts = pts.assign(mb_65=mb_65_plot.data[pts_index].astype(bool),
                 mb_15=mb_15_plot.data[pts_index].astype(bool),
                 uf=uf_plot.data[pts_index].astype(bool))

# export point lookup as csv
pts_dir = batch_dir + 'dem_r.25_points.csv'
pts.to_csv(pts_dir, index=False)

# format point ids as raster
id_raster = rastools.raster_load(dem_in)
id_raster.data = np.full([id_raster.rows, id_raster.cols], id_raster.no_data).astype(int)
id_raster.data[pts_index] = pts.id
# save id raster to file
id_raster_out = batch_dir + 'dem_r.25_point_ids.tif'
rastools.raster_save(id_raster, id_raster_out, data_format="int32")

# point subsets
pts_mb_65 = pts[pts.mb_65]
pts_dir = batch_dir + 'dem_r.25_points_mb_65.csv'
pts_mb_65.to_csv(pts_dir, index=False)

pts_mb_15 = pts[pts.mb_15]
pts_dir = batch_dir + 'dem_r.25_points_mb_15.csv'
pts_mb_15.to_csv(pts_dir, index=False)

pts_uf = pts[pts.uf]
pts_dir = batch_dir + 'dem_r.25_points_uf.csv'
pts_uf.to_csv(pts_dir, index=False)

# create cookie cutters of sites for each resolution
for rr in resolution:
    file_out = 'uf_plot_r' + rr + 'm.tif'
    site_poly = uf_poly
    template_in = template_scheme.replace('<RES>', rr)
    ras = rastools.raster_load(template_in)
    ras.data = np.full((ras.rows, ras.cols), 0)
    ras.no_data = 0
    ras_out = batch_dir + file_out
    rastools.raster_save(ras, ras_out, data_format='byte')
    rastools.raster_burn(ras_out, site_poly, 1)

for rr in resolution:
    file_out = 'uc_plot_r' + rr + 'm.tif'
    site_poly = uc_poly
    template_in = template_scheme.replace('<RES>', rr)
    ras = rastools.raster_load(template_in)
    ras.data = np.full((ras.rows, ras.cols), 0)
    ras.no_data = 0
    ras_out = batch_dir + file_out
    rastools.raster_save(ras, ras_out, data_format='byte')
    rastools.raster_burn(ras_out, site_poly, 1)

for rr in resolution:
    file_out = 'site_plots_r' + rr + 'm.tif'
    template_in = template_scheme.replace('<RES>', rr)
    ras = rastools.raster_load(template_in)
    ras.data = np.full((ras.rows, ras.cols), 0)
    ras.no_data = 0
    ras_out = batch_dir + file_out
    rastools.raster_save(ras, ras_out, data_format='uint16')
    rastools.raster_burn(ras_out, uf_poly, 1)
    rastools.raster_burn(ras_out, uc_poly, 2)

for rr in resolution:
    file_out = 'mb_15_plot_r' + rr + 'm.tif'
    site_poly = mb_15_poly
    template_in = template_scheme.replace('<RES>', rr)
    ras = rastools.raster_load(template_in)
    ras.data = np.full((ras.rows, ras.cols), 0)
    ras.no_data = 0
    ras_out = batch_dir + file_out
    rastools.raster_save(ras, ras_out, data_format='byte')
    rastools.raster_burn(ras_out, site_poly, 1)

for rr in resolution:
    file_out = 'mb_65_plot_r' + rr + 'm.tif'
    site_poly = mb_65_poly
    template_in = template_scheme.replace('<RES>', rr)
    ras = rastools.raster_load(template_in)
    ras.data = np.full((ras.rows, ras.cols), 0)
    ras.no_data = 0
    ras_out = batch_dir + file_out
    rastools.raster_save(ras, ras_out, data_format='byte')
    rastools.raster_burn(ras_out, site_poly, 1)

# create lookup table for 1m and .25m grid points