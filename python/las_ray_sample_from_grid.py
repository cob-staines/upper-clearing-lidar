import las_ray_sampling as lrs
import numpy as np
import pandas as pd
import os
import rastools

# config for batch rs_hemi

vox = lrs.VoxelObj()
vox.las_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\LAS\\19_149_las_proc_classified_merged.las'
# vox.las_in = 'C:\\Users\\jas600\\workzone\\data\\las\\19_149_las_proc_classified_merged.las'
vox.traj_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_all_traj.txt'
# vox.traj_in = 'C:\\Users\\jas600\\workzone\\data\\las\\19_149_all_traj.txt'
vox.return_set = 'first'
vox.drop_class = 7
hdf5_path = vox.las_in.replace('.las', '_ray_sampling_' + vox.return_set + '_returns_drop_' + str(vox.drop_class) + '_int.h5')
vox.hdf5_path = hdf5_path
vox.chunksize = 10000000
voxel_length = .25
vox.step = np.full(3, voxel_length)
vox.sample_length = voxel_length/np.pi
vox_id = 'rs_vl' + str(voxel_length)
vox.id = vox_id

# vox = lrs.las_to_vox(vox, np.uint16, create_new_hdf5=True)

# LOAD VOX
vox = lrs.vox_load(hdf5_path, vox_id)


batch_dir = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\batches\\lrs_mb_15_dem_.10m\\'
dem_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\DEM\\interpolated\\19_149_dem_interpolated_r.25m.tif"
mask_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\hemi_grid_points\\mb_65_1m\\mb_15_plot_r.25m.tif"

# batch_dir = 'C:\\Users\\jas600\\workzone\\data\\ray_sampling\\batches\\lrs_uf_1m\\'
# pts_in = 'C:\\Users\\jas600\\workzone\\data\\ray_sampling\\mb_65_1m\\1m_dem_points_uf.csv'

# # load points
# pts = pd.read_csv(pts_in)

rsgmeta = lrs.RaySampleGridMetaObj()
# self.id = None
# self.file_name = None

rsgmeta.agg_method = 'linear'

if rsgmeta.agg_method == 'nb_lookup':
    mean_path_length = 2 * np.pi / (6 + np.pi) * voxel_length  # mean path length through a voxel cube across angles (m)
    prior_weight = 5  # in units of scans (1 <=> equivalent weight to 1 expected voxel scan)
    prior_b = mean_path_length * prior_weight
    prior_a = prior_b * 0.01
    rsgmeta.prior = [prior_a, prior_b]
elif rsgmeta.agg_method == 'linear':
    samps = (vox.sample_data > 0)
    trans = vox.return_data[samps] // (vox.sample_data[samps] * vox.sample_length)
    prior = np.var(trans)
else:
    raise Exception('Aggregation method ' + rsgmeta.agg_method + ' unknown.')


rsgmeta.ray_sample_length = vox.sample_length
rsgmeta.ray_iterations = 100  # model runs for each ray, from which median and std of returns is calculated

# image dimensions
rsgmeta.phi = 0
rsgmeta.theta = 0

# image geometry
rsgmeta.max_distance = 50  # meters
rsgmeta.min_distance = voxel_length * np.sqrt(3)  # meters


# output file dir
rsgmeta.src_file = dem_in
rsgmeta.mask_file = mask_in
rsgmeta.file_dir = batch_dir + "outputs\\"
if not os.path.exists(rsgmeta.file_dir):
    os.makedirs(rsgmeta.file_dir)

file_out = 'rs_mb_15_r.25_p{:.4f}_t{:.4f}_linear.tif'.format(rsgmeta.phi, rsgmeta.theta)

# phi = rsgmeta.phi
# theta = rsgmeta.theta
# min_dist = rsgmeta.min_distance
# max_dist = rsgmeta.max_distance
rays_in = lrs.dem_to_rays(dem_in, vox, rsgmeta.phi, rsgmeta.theta, mask_in, rsgmeta.min_distance, rsgmeta.max_distance)
# chunksize = 100000
# agg_sample_length = rsgmeta.ray_sample_length
# prior = rsgmeta.prior
# ray_iterations = rsgmeta.ray_iterations
# method = 'linear'
# commentation = True
rays_out = lrs.agg_chunk(100000, vox, rays_in, rsgmeta.ray_sample_length, rsgmeta.prior, rsgmeta.ray_iterations, rsgmeta.agg_method, commentation=True)
ras = lrs.ray_stats_to_dem(rays_out, dem_in, rsgmeta.file_dir + file_out)

# ras = ray_stats_to_dem(rays_out, dem_in)
# rastools.raster_save(ras, ras_out)

# rshm = lrs.rs_hemigen(rshmeta, vox, initial_index=0)
# rshm = lrs.rs_pts(rshmeta, vox, initial_index)

###
#
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# import tifffile as tif
# ii = 0
# peace = tif.imread(rshm.file_dir[ii] + rshm.file_name[ii])
# plt.imshow(peace[:, :, 2], interpolation='nearest')