import las_ray_sampling as lrs
import numpy as np
import pandas as pd
import os

# config for batch rs_hemi

vox = lrs.VoxelObj()
vox.las_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\LAS\\19_149_las_proc_classified_merged.las'
# vox.las_in = 'C:\\Users\\jas600\\workzone\\data\\las\\19_149_las_proc_classified_merged.las'
vox.traj_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_all_traj.txt'
# vox.traj_in = 'C:\\Users\\jas600\\workzone\\data\\las\\19_149_all_traj.txt'
vox.return_set = 'first'
vox.drop_class = 7
hdf5_path = vox.las_in.replace('.las', '_ray_sampling_' + vox.return_set + '_returns_drop_' + str(vox.drop_class) + '.h5')
vox.hdf5_path = hdf5_path
vox.chunksize = 10000000
voxel_length = 1
vox.step = np.full(3, voxel_length)
vox.sample_length = voxel_length/np.pi
vox_id = 'rs_vl' + str(voxel_length)
vox.id = vox_id

# vox = lrs.ray_sample_las(vox, create_new_hdf5=True)

# LOAD VOX
vox = lrs.vox_load(hdf5_path, vox_id)


batch_dir = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\batches\\lrs_uf_1m\\'
pts_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\hemi_grid_points\\mb_65_1m\\1m_dem_points_uf.csv"


# batch_dir = 'C:\\Users\\jas600\\workzone\\data\\hemigen\\mb_15_1m_pr.15_os10\\'
# las_in = "C:\\Users\\jas600\\workzone\\data\\hemigen\\hemi_lookups\\19_149_las_proc_classified_merged.las"
# pts_in = 'C:\\Users\\jas600\\workzone\\data\\hemigen\\hemi_lookups\\1m_dem_points_mb_15.csv'

# load points
pts = pd.read_csv(pts_in)

pts = pts[0:10]

# calculate expected points and varience
ratio = .05  # ratio of voxel area weight of prior
F = .16 * 0.05  # expected footprint area
V = np.prod(vox.step)  # volume of each voxel

# gamma prior hyperparameters
prior_b = ratio * V / F  # path length required to scan "ratio" of one voxel volume
prior_a = prior_b * 0.001  # prior_b * K / N

# build meta object
rshmeta = lrs.RaySampleHemiMetaObj()
rshmeta.prior = [prior_a, prior_b]
rshmeta.ray_sample_length = vox.sample_length
rshmeta.ray_iterations = 100
rshmeta.max_phi_rad = np.pi/2

# output file dir
rshmeta.file_dir = batch_dir + "outputs\\"
if not os.path.exists(rshmeta.file_dir):
    os.makedirs(rshmeta.file_dir)

# max distance of points considered in image
rshmeta.max_distance = 50  # meters
rshmeta.min_distance = .5  # meters
hemi_m_above_ground = 0  # meters

# image size
rshmeta.img_size = 100  # square, in pixels/ray samples


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