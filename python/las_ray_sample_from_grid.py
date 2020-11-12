import las_ray_sampling as lrs
import numpy as np
import pandas as pd
import os

vox = lrs.VoxelObj()
vox.las_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\LAS\\19_149_las_proc_classified_merged.las'
# vox.las_in = 'C:\\Users\\jas600\\workzone\\data\\las\\19_149_las_proc_classified_merged.las'
vox.traj_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_all_traj.txt'
# vox.traj_in = 'C:\\Users\\jas600\\workzone\\data\\las\\19_149_all_traj.txt'
vox.return_set = 'first'
vox.drop_class = 7
vox.las_traj_hdf5 = vox.las_in.replace('.las', '_ray_sampling_' + vox.return_set + '_returns_drop_' + str(vox.drop_class) + '_las_traj.h5')
vox.sample_precision = np.uint32
vox.return_precision = np.uint32
vox.las_traj_chunksize = 10000000
vox.cw_rotation = -34 * np.pi / 180
voxel_length = .25
vox.step = np.full(3, voxel_length)
vox.sample_length = voxel_length/np.pi
vox.vox_hdf5 = vox.las_in.replace('.las', '_ray_sampling_' + vox.return_set + '_returns_drop_' + str(vox.drop_class) + '_r' + str(voxel_length) + 'm_vox.h5')


# vox = lrs.las_to_vox(vox, run_las_traj=False, fail_overflow=False)

# # LOAD VOX
print('Loading vox... ', end='')
vox = lrs.load_vox_meta(vox.vox_hdf5, load_data=True)
print('done')


batch_dir = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_mb_15_dem_.10m\\'
dem_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\DEM\\interpolated\\19_149_dem_interpolated_r.25m.tif"
mask_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\hemi_grid_points\\mb_65_1m\\mb_15_plot_r.25m.tif"


rsgmeta = lrs.RaySampleGridMetaObj()

rsgmeta.agg_method = 'beta'

print('Calculating prior... ', end='')
if rsgmeta.agg_method == 'nb_lookup':
    mean_path_length = 2 * np.pi / (6 + np.pi) * voxel_length  # mean path length through a voxel cube across angles (m)
    prior_weight = 5  # in units of scans (1 <=> equivalent weight to 1 expected voxel scan)
    prior_b = mean_path_length * prior_weight
    prior_a = prior_b * 0.01
    rsgmeta.prior = [prior_a, prior_b]
elif rsgmeta.agg_method == 'linear':
    samps = (vox.sample_data > 0)
    trans = vox.return_data[samps] // (vox.sample_data[samps] * vox.sample_length)
    rsgmeta.prior = np.var(trans)
elif rsgmeta.agg_method == 'beta':
    val = (vox.sample_data > 0)  # roughly 50% at .25m
    rate = vox.return_data[val] / vox.sample_data[val]
    mu = np.mean(rate)
    sig2 = np.var(rate)

    alpha = ((1 - mu)/sig2 - 1/mu) * (mu ** 2)
    beta = alpha * (1/mu - 1)
    rsgmeta.prior = [alpha, beta]
else:
    raise Exception('Aggregation method ' + rsgmeta.agg_method + ' unknown.')
print('done')


rsgmeta.ray_sample_length = vox.sample_length
# rsgmeta.ray_iterations = 100  # model runs for each ray, from which median and std of returns is calculated

# ray geometry
phi_step = (np.pi / 2) / (180 * 2)
rsgmeta.set_phi_size = 61  # square, in pixels/ray samples
rsgmeta.set_max_phi_rad = phi_step * rsgmeta.set_phi_size
# rsgmeta.set_max_phi_rad = np.pi/2
# ray m above ground?
rsgmeta.max_distance = 50  # meters
rsgmeta.min_distance = voxel_length * np.sqrt(3)  # meters

# output file dir
rsgmeta.src_ras_file = dem_in
rsgmeta.mask_file = mask_in
rsgmeta.file_dir = batch_dir + "outputs\\"
if not os.path.exists(rsgmeta.file_dir):
    os.makedirs(rsgmeta.file_dir)


# calculate hemisphere of phi and theta values
rsgmeta.phi = 0
rsgmeta.theta = 0

# export phi_theta_lookup of vectors in grid
vector_set = lrs.hemi_vectors(rsgmeta.set_phi_size, rsgmeta.set_max_phi_rad).sort_values('phi').reset_index(drop=True)
vector_set.to_csv(rsgmeta.file_dir + "phi_theta_lookup.csv", index=False)

rsgmeta.id = vector_set.index.values
rsgmeta.phi = vector_set.phi.values
rsgmeta.theta = vector_set.theta.values
rsgmeta.file_name = ["las_19_149_rs_mb_15_r.25_p{:.4f}_t{:.4f}.tif".format(rsgmeta.phi[ii], rsgmeta.theta[ii]) for ii in rsgmeta.id]

rsgm = lrs.rs_gridgen(rsgmeta, vox, initial_index=0)

##


# min_dist = rsgmeta.min_distance
# max_dist = rsgmeta.max_distance



# ###
# #
# # import matplotlib
# # matplotlib.use('TkAgg')
# # import matplotlib.pyplot as plt
# # import tifffile as tif
# # ii = 0
# # peace = tif.imread(rshm.file_dir[ii] + rshm.file_name[ii])
# # plt.imshow(peace[:, :, 2], interpolation='nearest')