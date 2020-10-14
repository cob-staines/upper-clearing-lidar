import pandas as pd
import numpy as np
import laslib
import rastools
import os

# build point list from DEM
template_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\TEMPLATES\\19_149_all_point_density_r1.00m.bil'
dem_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\DEM\\interpolated\\19_149_dem_r1.00m_q0.25_interpolated_t1.tif'
mb_65_poly = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\site_library\\mb_65_poly.shp'
mb_15_poly = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\site_library\\mb_15_poly.shp'
uf_poly = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\site_library\\upper_forest_poly_UTM11N.shp'
batch_dir = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\hemi_grid_points\\mb_65_1m\\'

# dem_in = 'C:\\Users\\jas600\\workzone\\data\\hemigen\\hemi_lookups\\19_149_dem_r1.00m_q0.25_interpolated_min1.tif'
# site_poly = 'C:\\Users\\jas600\\workzone\\data\\hemigen\\hemi_lookups\\upper_forest_poly_UTM11N.shp'
# batch_dir = 'C:\\Users\\jas600\\workzone\\data\\hemigen\\uf_1m_pr_0_os_0.5\\'

# create batch dir if does not exist
if not os.path.exists(batch_dir):
    os.makedirs(batch_dir)

# load dem sampled to raster template
ddict = {'n_count': template_in,
         'z_m': dem_in}
pts = rastools.pd_sample_raster_gdal(ddict, include_nans=True)

# add point id
pts = pts.reset_index()
pts.columns = ['id', 'x_utm11n', 'y_utm11n', 'x_index', 'y_index', 'n_count', 'z_m']


# # add flag for mb_65
# load template
site_plot = rastools.raster_load(template_in)
# fill data with zeros
site_plot.data = np.full((site_plot.rows, site_plot.cols), 0)
# save to file
mb_65_plot_dir = batch_dir + 'mb_65_plot_over_dem.tiff'
rastools.raster_save(site_plot, mb_65_plot_dir, data_format='byte')
# burn site polygon into plot data as ones
rastools.raster_burn(mb_65_plot_dir, mb_65_poly, 1)
# load plot data
mb_65_plot = rastools.raster_load(mb_65_plot_dir)

# # add flag for mb_65
# load template
site_plot = rastools.raster_load(template_in)
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
site_plot = rastools.raster_load(template_in)
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
pts = pts.assign(mb_65=mb_65_plot.data[pts_index].astype(bool), mb_15=mb_15_plot.data[pts_index].astype(bool), uf=uf_plot.data[pts_index].astype(bool))

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
pts_mb_65 = pts[pts.mb_65]
pts_dir = batch_dir + '1m_dem_points_mb_65.csv'
pts_mb_65.to_csv(pts_dir, index=False)

pts_mb_15 = pts[pts.mb_15]
pts_dir = batch_dir + '1m_dem_points_mb_15.csv'
pts_mb_15.to_csv(pts_dir, index=False)

pts_uf = pts[pts.uf]
pts_dir = batch_dir + '1m_dem_points_uf.csv'
pts_uf.to_csv(pts_dir, index=False)

# create cookie cutters of sites at different resolutions
nodata = 0
resolution = ['.10', '.25', '.50', '1.00']
for rr in resolution:
    template_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\TEMPLATES\\19_149_all_point_density_r' + rr + 'm.bil'
    uf_rp = rastools.raster_load(template_in)
    uf_rp.data = rastools.gdal_raster_reproject(uf_plot_dir, template_in, nodatavalue=nodata)[:, :, 0]
    uf_rp.no_data = nodata
    cc_out = uf_plot_dir.replace('.tiff', '_cookiecutter_r' + rr + 'm.tiff')
    rastools.raster_save(uf_rp, cc_out, data_format='byte')

for rr in resolution:
    template_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\TEMPLATES\\19_149_all_point_density_r' + rr + 'm.bil'
    mb_15_rp = rastools.raster_load(template_in)
    mb_15_rp.data = rastools.gdal_raster_reproject(mb_15_plot_dir, template_in, nodatavalue=nodata)[:, :, 0]
    mb_15_rp.no_data = nodata
    cc_out = mb_15_plot_dir.replace('.tiff', '_cookiecutter_r' + rr + 'm.tiff')
    rastools.raster_save(mb_15_rp, cc_out, data_format='byte')

for rr in resolution:
    template_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\TEMPLATES\\19_149_all_point_density_r' + rr + 'm.bil'
    mb_65_rp = rastools.raster_load(template_in)
    mb_65_rp.data = rastools.gdal_raster_reproject(mb_65_plot_dir, template_in, nodatavalue=nodata)[:, :, 0]
    mb_65_rp.no_data = nodata
    cc_out = mb_65_plot_dir.replace('.tiff', '_cookiecutter_r' + rr + 'm.tiff')
    rastools.raster_save(mb_65_rp, cc_out, data_format='byte')