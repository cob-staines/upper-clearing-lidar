from libraries import raslib
import numpy as np
import os

angle = 15
doy_set = ["19_149"]

for doy in doy_set:

    # load components
    dir_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\%DOY%\\%DOY%_las_proc\\OUTPUT_FILES\\RAS\\".replace("%DOY%", doy)
    # FG_in = "%DOY%_first_ground_point_density_a%ANGLE%_r.10m.bil".replace('%ANGLE%', str(angle)).replace("%DOY%", doy)
    # LG_in = "%DOY%_last_ground_point_density_a%ANGLE%_r.10m.bil".replace('%ANGLE%', str(angle)).replace("%DOY%", doy)
    # FC_in = "%DOY%_first_veg_point_density_a%ANGLE%_r.10m.bil".replace('%ANGLE%', str(angle)).replace("%DOY%", doy)
    FG_in = "%DOY%_first_ground_point_density_a%ANGLE%_r.25m.bil".replace('%ANGLE%', str(angle)).replace("%DOY%", doy)
    LG_in = "%DOY%_last_ground_point_density_a%ANGLE%_r.25m.bil".replace('%ANGLE%', str(angle)).replace("%DOY%", doy)
    FC_in = "%DOY%_first_veg_point_density_a%ANGLE%_r.25m.bil".replace('%ANGLE%', str(angle)).replace("%DOY%", doy)

    # outputs
    dir_out = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\%DOY%\\%DOY%_las_proc\\OUTPUT_FILES\\LPM\\".replace("%DOY%", doy)
    lpmf_out = "%DOY%_LPM-first_a%ANGLE%_r0.10m.tif".replace('%ANGLE%', str(angle)).replace("%DOY%", doy)
    fcov_out = "%DOY%_fcov_a%ANGLE%_r0.10m.tif".replace('%ANGLE%', str(angle)).replace("%DOY%", doy)
    lpml_out = "%DOY%_LPM-last_a%ANGLE%_r0.10m.tif".replace('%ANGLE%', str(angle)).replace("%DOY%", doy)
    lpmc_out = "%DOY%_LPM-canopy_a%ANGLE%_r0.10m.tif".replace('%ANGLE%', str(angle)).replace("%DOY%", doy)

    # load raster data in
    FG = raslib.raster_load(dir_in + FG_in)
    LG = raslib.raster_load(dir_in + LG_in)
    FC = raslib.raster_load(dir_in + FC_in)

    # store no-data value
    no_data = FG.no_data

    # set no_data points to np.nan
    FG.data[FG.data == no_data] = 0
    LG.data[LG.data == no_data] = 0
    FC.data[FC.data == no_data] = 0

    # do calculations
    lpmf = raslib.raster_load(dir_in + FG_in)
    num = FG.data
    denom = (FG.data + FC.data)
    denom[denom == 0] = np.nan
    lpmf.data = num/denom

    lpml = raslib.raster_load(dir_in + FG_in)
    num = (LG.data + FG.data)
    denom = (LG.data + FG.data + FC.data)
    denom[denom == 0] = np.nan
    lpml.data = num/denom

    lpmc = raslib.raster_load(dir_in + FG_in)
    num = LG.data
    denom = (LG.data + FC.data)
    denom[denom == 0] = np.nan
    lpmc.data = num/denom

    fcov = raslib.raster_load(dir_in + FG_in)
    fcov.data = 1 - lpmf.data

    lpmf.data[np.isnan(lpmf.data)] = no_data
    lpml.data[np.isnan(lpml.data)] = no_data
    lpmc.data[np.isnan(lpmc.data)] = no_data
    fcov.data[np.isnan(fcov.data)] = no_data



    # export
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    raslib.raster_save(lpmf, dir_out + lpmf_out)
    raslib.raster_save(lpml, dir_out + lpml_out)
    raslib.raster_save(lpmc, dir_out + lpmc_out)
    raslib.raster_save(fcov, dir_out + fcov_out)
