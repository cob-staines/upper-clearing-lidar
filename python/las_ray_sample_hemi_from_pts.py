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
vox.vox_hdf5 = vox.las_in.replace('.las', '_ray_sampling_' + vox.return_set + '_returns_drop_' + str(vox.drop_class) + '_vox_rot.h5')
vox.sample_precision = np.uint32
vox.return_precision = np.uint32
vox.las_traj_chunksize = 10000000
vox.cw_rotation = -34 * np.pi / 180
voxel_length = .25
vox.step = np.full(3, voxel_length)
vox.sample_length = voxel_length/np.pi


# vox = lrs.las_to_vox(vox, run_las_traj=False, fail_overflow=False)

# # LOAD VOX
vox = lrs.load_vox_meta(vox.vox_hdf5, load_data=False)


# batch_dir = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\batches\\lrs_uf_1m\\'
# pts_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\hemi_grid_points\\mb_65_1m\\1m_dem_points_uf.csv"


batch_dir = 'C:\\Users\\jas600\\workzone\\data\\ray_sampling\\batches\\lrs_uf_1m\\'
pts_in = 'C:\\Users\\jas600\\workzone\\data\\ray_sampling\\mb_65_1m\\1m_dem_points_uf.csv'

# load points
pts = pd.read_csv(pts_in)


rshmeta = lrs.RaySampleHemiMetaObj()

# ray resampling parameters
# ratio = .05  # ratio of voxel area weight of prior
# F = .16 * 0.05  # expected footprint area
# V = np.prod(vox.step)  # volume of each voxel
mean_path_length = 2 * np.pi / (6 + np.pi) * voxel_length  # mean path length through a voxel cube across angles (m)
prior_weight = 5  # in units of scans (1 <=> equivalent weight to 1 expected voxel scan)
# prior_b = ratio * V / F  # path length required to scan "ratio" of one voxel volume
prior_b = mean_path_length * prior_weight
prior_a = prior_b * 0.01
rshmeta.prior = [prior_a, prior_b]
rshmeta.ray_sample_length = vox.sample_length
rshmeta.ray_iterations = 100  # model runs for each ray, from which median and std of returns is calculated

# image dimensions
phi_step = (np.pi/2) / (180 * 2)
rshmeta.img_size = 61  # square, in pixels/ray samples
rshmeta.max_phi_rad = phi_step * rshmeta.img_size

# image geometry
hemi_m_above_ground = 0  # meters
rshmeta.max_distance = 50  # meters
rshmeta.min_distance = voxel_length * np.sqrt(3)  # meters


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

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tifffile as tif
ii = 0
peace = tif.imread(rshm.file_dir[ii] + rshm.file_name[ii])
plt.imshow(peace[:, :, 2], interpolation='nearest')