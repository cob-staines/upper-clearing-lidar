import numpy as np
from scipy.ndimage import convolve
from libraries import raslib

# config
ras_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\CHM\\19_149_spike_free_chm_r.25m.tif"
canopy_min_elev = 1.25
# canopy_min_elev = 10
kernel_dim = 3  # for smoothing
step_size = 0.5  # for vector search walk, units of raster
max_step = 400  # max number of steps
initial_dist = 3  # initial distance from point (in raster cells)
raster_length = .25  # m
angle_count = 192
smooth_cutoff = 0.9
out_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\moeser\\"

# load raster
ras = raslib.raster_load(ras_in)

# define canopy binary
canopy = np.full([ras.rows, ras.cols], 0)
canopy[ras.data >= canopy_min_elev] = 1

# smoothe
kernel = np.full([kernel_dim, kernel_dim], 1)
convolved = convolve(canopy, kernel) / np.sum(kernel)

# points to search:
ids_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\site_library\\hemi_grid_points\\mb_65_r.25m_snow_off_offset.25\\dem_r.25_point_ids.tif'
ddict = {'chm': ras_in,
         'lrs_id': ids_in,
         'uf': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\site_library\\hemi_grid_points\\mb_65_r.25m_snow_off_offset0\\uf_plot_r.25m.tif',
         }
df = raslib.pd_sample_raster_gdal(ddict, include_nans=False, mode="nearest")

# filter to uf only... for now....
df = df.loc[df.uf == 1, :]

can_dist = np.full((len(df), angle_count), np.nan)

for ii in range(len(df)):
    coords = np.array([df.y_index.iloc[ii], df.x_index.iloc[ii]])
    for jj in range(angle_count):
        aa = jj * 2 * np.pi / angle_count
        displacement = np.array([np.cos(aa), np.sin(aa)])
        hit = False
        step_count = 0

        while ~hit & (step_count < max_step):
            dd = initial_dist + step_count * step_size
            test_pnt = np.floor(coords + dd * displacement).astype(int)
            test = convolved[(test_pnt[0], test_pnt[1])]
            if test >= smooth_cutoff:
                # record as hit
                hit = True
                can_dist[ii, jj] = dd
            step_count += 1

    print(ii)

can_dist[np.isnan(can_dist)] = initial_dist + max_step * step_size

can_dist = can_dist / raster_length

# calculate metrics!
df.loc[:, "min_dist_to_canopy"] = np.nanmin(can_dist, axis=1)
df.loc[:, "mean_dist_to_canopy"] = np.nanmean(can_dist, axis=1)
df.loc[:, "median_dist_to_canopy"] = np.nanmedian(can_dist, axis=1)
df.loc[:, "total_gap_area"] = np.pi * np.nansum(can_dist ** 2, axis=1) / angle_count

# save as rasters
colname = "total_gap_area"
template_in = ras_in

def pd_to_ras(df, colname, template_in, file_out):
    temp = raslib.raster_load(template_in)
    temp.data[:, :] = temp.no_data
    temp.data[df.y_index, df.x_index] = df.loc[:, colname]

    raslib.raster_save(temp, file_out)


pd_to_ras(df, "min_dist_to_canopy", ras_in, out_dir + "uf_min_distance_to_canopy_" + str(canopy_min_elev) + "m.tif")
pd_to_ras(df, "mean_dist_to_canopy", ras_in, out_dir + "uf_mean_distance_to_canopy_" + str(canopy_min_elev) + "m.tif")
pd_to_ras(df, "median_dist_to_canopy", ras_in, out_dir + "uf_median_distance_to_canopy_" + str(canopy_min_elev) + "m.tif")
pd_to_ras(df, "total_gap_area", ras_in, out_dir + "uf_total_gap_area_" + str(canopy_min_elev) + "m.tif")

###