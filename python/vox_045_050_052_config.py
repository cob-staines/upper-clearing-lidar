import numpy as np
import las_ray_sampling as lrs
config_id = "045_050_052_r0.25"

voxList = ['C:\\Users\\jas600\\workzone\\data\\ray_sampling\\sources\\19_045_all_WGS84_utm11N_19_045_r0.25_vox.h5',
           'C:\\Users\\jas600\\workzone\\data\\ray_sampling\\sources\\19_050_all_WGS84_utm11N_19_050_r0.25_vox.h5',
           'C:\\Users\\jas600\\workzone\\data\\ray_sampling\\sources\\19_052_all_WGS84_utm11N_19_052_r0.25_vox.h5']

vox_out = 'C:\\Users\\jas600\\workzone\\data\\ray_sampling\\sources\\045_050_052_combined_WGS84_utm11N_r0.25_vox.h5'
# vox_out = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\sources\\045_050_052_combined_WGS84_utm11N_r0.25_vox.h5'
z_slices = 8

# BUILD VOX
# combine voxel spaces
lrs.vox_addition(voxList, vox_out, z_slices)
vox = lrs.load_vox_meta(vox_out, load_data=False, load_post=False)
# calculate prior
lrs.beta_lookup_prior_calc(vox, z_slices, agg_sample_length=None)
# combine voxel spaces
vox = lrs.load_vox_meta(vox_out, load_data=False, load_post=True)
