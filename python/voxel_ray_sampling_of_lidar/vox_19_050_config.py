import numpy as np
from libraries import las_ray_sampling as lrs
config_id = "19_050_r0.25"

vox = lrs.VoxelObj()
# vox.las_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\LAS\\19_149_las_proc_classified_merged.las'
vox.las_in = 'C:\\Users\\jas600\\workzone\\data\\ray_sampling\\sources\\19_050_all_WGS84_utm11N.las'
# vox.traj_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_all_traj.txt'
vox.traj_in = 'C:\\Users\\jas600\\workzone\\data\\ray_sampling\\sources\\19_050_all_trajectories_WGS84_utm11N.txt'
vox.return_set = 'first'
vox.drop_class = 7
vox.las_traj_hdf5 = vox.las_in.replace('.las', '_ray_sampling_' + vox.return_set + '_returns_drop_' + str(vox.drop_class) + '_las_traj.h5')
vox.sample_dtype = np.uint32
vox.return_dtype = np.uint32
vox.las_traj_chunksize = 5000000
vox.cw_rotation = -34 * np.pi / 180
voxel_length = .25
vox.step = np.full(3, voxel_length)
vox.sample_length = voxel_length / np.pi
vox.vox_hdf5 = vox.las_in.replace('.las', "_" + config_id + '_vox.h5')

# specify origin and max for alignment with other sets (pulled from 19_149 vox)
vox.origin = np.array([-2.63693299e+06,  5.03236474e+06,  1.80983250e+03])
vox.max = np.array([-2.63665114e+06,  5.03262022e+06,  1.87498250e+03])

z_slices = 8

# # BUILD VOX
# vox = lrs.las_to_vox(vox, z_slices, run_las_traj=True, fail_overflow=False, calc_prior=False)

# # LOAD VOX
vox = lrs.load_vox(vox.vox_hdf5, load_data=False)
