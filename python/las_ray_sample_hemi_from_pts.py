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

z_slices = 4
# vox = lrs.las_to_vox(vox, z_slices, run_las_traj=False, fail_overflow=False)


# # LOAD VOX
print('Loading vox... ', end='')
vox = lrs.load_vox_meta(vox.vox_hdf5, load_data=True)
print('done')


batch_dir = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\batches\\lrs_uf_1m\\'
pts_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\hemi_grid_points\\mb_65_1m\\1m_dem_points_uf.csv"


# batch_dir = 'C:\\Users\\jas600\\workzone\\data\\ray_sampling\\batches\\lrs_uf_1m\\'
# pts_in = 'C:\\Users\\jas600\\workzone\\data\\ray_sampling\\mb_65_1m\\1m_dem_points_uf.csv'

# load points
pts = pd.read_csv(pts_in)


rshmeta = lrs.RaySampleGridMetaObj()

rshmeta.agg_method = 'beta'

print('Calculating prior... ', end='')
if rshmeta.agg_method == 'nb_lookup':
    mean_path_length = 2 * np.pi / (6 + np.pi) * voxel_length  # mean path length through a voxel cube across angles (m)
    prior_weight = 5  # in units of scans (1 <=> equivalent weight to 1 expected voxel scan)
    prior_b = mean_path_length * prior_weight
    prior_a = prior_b * 0.01
    rshmeta.prior = [prior_a, prior_b]
elif rshmeta.agg_method == 'linear':
    samps = (vox.sample_data > 0)
    trans = vox.return_data[samps] // (vox.sample_data[samps] * vox.sample_length)
    rshmeta.prior = np.var(trans)
elif rshmeta.agg_method == 'beta':
    val = (vox.sample_data > 0)  # roughly 50% at .25m
    rate = vox.return_data[val] / vox.sample_data[val]
    mu = np.mean(rate)
    sig2 = np.var(rate)

    alpha = ((1 - mu)/sig2 - 1/mu) * (mu ** 2)
    beta = alpha * (1/mu - 1)
    rshmeta.prior = [alpha, beta]
else:
    raise Exception('Aggregation method ' + rshmeta.agg_method + ' unknown.')
print('done')


rshmeta.ray_sample_length = vox.sample_length
# rsgmeta.ray_iterations = 100  # model runs for each ray, from which median and std of returns is calculated

# ray geometry
# phi_step = (np.pi / 2) / (180 * 2)
rshmeta.img_size = 61  # square, in pixels/ray samples
# rshmeta.max_phi_rad = phi_step * rshmeta.img_size
rshmeta.max_phi_rad = np.pi/2
hemi_m_above_ground = 0  # meters
rshmeta.max_distance = 50  # meters
rshmeta.min_distance = voxel_length * np.sqrt(3)  # meters

# rshmeta.ray_iterations = 100


# output file dir
rshmeta.file_dir = batch_dir + "outputs\\"
if not os.path.exists(rshmeta.file_dir):
    os.makedirs(rshmeta.file_dir)

rshmeta.id = pts.id
rshmeta.origin = np.array([pts.x_utm11n,
                           pts.y_utm11n,
                           pts.z_m + hemi_m_above_ground]).swapaxes(0, 1)

rshmeta.file_name = ["las_19_149_id_" + str(id) + ".tif" for id in pts.id]

rshm = lrs.rs_hemigen(rshmeta, vox, initial_index=0)

###
#
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# import tifffile as tif
# ii = 0
# peace = tif.imread(rshm.file_dir[ii] + rshm.file_name[ii])
# plt.imshow(peace[:, :, 2], interpolation='nearest')