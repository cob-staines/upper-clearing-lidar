import numpy as np
from libraries.las_ray_sampling import las_to_vox, load_vox, VoxelObj
config_id = "19_149_r0.25"

vox = VoxelObj()
# vox.las_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\LAS\\19_149_las_proc_classified_merged.las'
vox.las_in = 'C:\\Users\\jas600\\workzone\\data\\ray_sampling\\19_149_las_proc_classified_merged.las'
# vox.traj_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_all_traj.txt'
vox.traj_in = 'C:\\Users\\jas600\\workzone\\data\\ray_sampling\\19_149_all_traj.txt'
vox.return_set = 'first'
vox.drop_class = 7
vox.las_traj_hdf5 = vox.las_in.replace('.las', '_ray_sampling_' + vox.return_set + '_returns_drop_' + str(vox.drop_class) + '_las_traj.h5')
vox.sample_dtype = np.uint32
vox.return_dtype = np.uint32
vox.las_traj_chunksize = 10000000
vox.cw_rotation = -34 * np.pi / 180
voxel_length = .25
vox.step = np.full(3, voxel_length)
vox.sample_length = voxel_length / np.pi
vox.vox_hdf5 = vox.las_in.replace('.las', config_id + '_r' + str(voxel_length) + '_vox.h5')

z_slices = 4

# # BUILD VOX
vox = las_to_vox(vox, z_slices, run_las_traj=True, fail_overflow=False, posterior_calc=False)

# # LOAD VOX
vox = load_vox(vox.vox_hdf5, load_data=False)

# # generate resample point cloud
# las_out = vox.vox_hdf5.replace(".h5", "_resampled.las")
# lrs.vox_to_las(vox.vox_hdf5, las_out, samps_per_vox=50, sample_threshold=1)
