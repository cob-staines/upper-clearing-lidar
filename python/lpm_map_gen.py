import rastools
import numpy as np
import os

# load components
dir_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\RAS\\"
FG_in = "19_149_las_proc_627975_5646450_first_ground_point_density_d30_.25m.bil"
LG_in = "19_149_las_proc_627975_5646450_last_ground_point_density_d30_.25m.bil"
FC_in = "19_149_las_proc_627975_5646450_first_veg_point_density_d30_.25m.bil"

# outputs
dir_out = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\LPM\\"
lpmf_out = "19_149_LPM-first_d30_0.25m.tif"
lpml_out = "19_149_LPM-last_d30_0.25m.tif"
lpmc_out = "19_149_LPM-canopy_d30_0.25m.tif"

FG = rastools.raster_load(dir_in + FG_in)
LG = rastools.raster_load(dir_in + LG_in)
FC = rastools.raster_load(dir_in + FC_in)

no_data = FG.no_data

FG.data[FG.data == no_data] = 0
LG.data[LG.data == no_data] = 0
FC.data[FC.data == no_data] = 0

# do calculations
lpmf = rastools.raster_load(dir_in + FG_in)
num = FG.data
denom = (FG.data + FC.data)
denom[denom == 0] = np.nan
lpmf.data = num/denom

lpml = rastools.raster_load(dir_in + FG_in)
num = (LG.data + FG.data)
denom = (LG.data + FG.data + FC.data)
denom[denom == 0] = np.nan
lpml.data.data = num/denom

lpmc = rastools.raster_load(dir_in + FG_in)
num = LG.data
denom = (LG.data + FC.data)
denom[denom == 0] = np.nan
lpmc.data = num/denom

lpmf.data[np.isnan(lpmf.data)] = no_data
lpml.data[np.isnan(lpml.data)] = no_data
lpmc.data[np.isnan(lpmc.data)] = no_data

# export
if not os.path.exists(dir_out):
    os.makedirs(dir_out)

rastools.raster_save(lpmf, dir_out + lpmf_out)
rastools.raster_save(lpml, dir_out + lpml_out)
rastools.raster_save(lpmc, dir_out + lpmc_out)
