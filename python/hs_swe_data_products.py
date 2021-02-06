import rastools
import numpy as np
import pandas as pd
import os

snow_on = ["19_045", "19_050", "19_052", "19_107", "19_123"]
snow_off = ["19_149"]
all_dates = snow_on + snow_off

resolution = [".05", ".10", ".25", "1.00"]


depth_regression = 'exp'
# bb = 1 / 0.0004215
# cc = 1 / 0.0657602
# aa = 67.92 + bb / cc


# # each day
# cc = np.array([0.267892, 0.0392898, 0.01628000, 1.000000, 0.2004976])
# bb = np.array([0.001987, 0.0002041, 0.00005863, 0.004345, 0.0007910])

# merged 50-52
cc = np.array([0.267892, 0.0262895130, 0.0262895130, 1.000000, 0.2004976])
bb = np.array([0.001987, 0.0001192949, 0.0001192949, 0.004345, 0.0007910])

cc = 1.0 / cc
bb = 1.0 / bb
aa = 67.92 + bb / cc

aa = dict(zip(snow_on, aa))
bb = dict(zip(snow_on, bb))
cc = dict(zip(snow_on, cc))


# depth_regression = 'density'
#
# # all veg, each day, depth-density
# depth_to_density_intercept = dict(zip(snow_on, [183.5431, 110.2249, 72.5015, 224.6406, 223.5683]))
# depth_to_density_slope = dict(zip(snow_on, 100*np.array([0.1485, 1.2212, 1.5346, 1.7833, 1.2072])))

# # clearing, each day, depth-density
# depth_to_density_intercept = dict(zip(snow_on, [109.1403, 79.0724, 75.2462, 284.3746, 291.7717]))
# depth_to_density_slope = dict(zip(snow_on, 100*np.array([1.2717, 1.6568, 1.4417, 0.5656, 0.1317])))

# # all veg, days 1-3 combined, depth-density
# depth_to_density_intercept = dict(zip(snow_on, [120.248, 120.248, 120.248, 120.248, 120.248]))
# depth_to_density_slope = dict(zip(snow_on, 100*np.array([1.029, 1.029, 1.029, 1.029, 1.029])))

# depth_regression = 'swe'
#
# # forest only, each day, depth-SWE
# depth_to_swe_slope = dict(zip(snow_on, 100*np.array([2.695, 2.7394, 3.0604, 3.1913, 2.5946])))
# # all, each day, depth-SWE
# depth_to_swe_slope = dict(zip(snow_on, 100*np.array([2.695, 2.7394, 3.0604, 3.1913, 2.5946])))


# ceiling_depths = [0.87, 0.96, 0.97, 0.93, 0.98]  # this is not very pretty... can I make this somehow cleaner?

hs_clean_ceiling_res = resolution[0]  # use smallest resolution
hs_clean_ceiling_quantile = 0.999  # determined visually...

# las_in_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\<DATE>\\<DATE>_las_proc\\OUTPUT_FILES\\LAS\\<DATE>_las_proc_classified_merged_ground.las'
# dem_ref_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc_xx\\OUTPUT_FILES\\TEMPLATES\\19_149_all_point_density_r<RES>m.bil'
# dem_dir_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\<DATE>\\<DATE>_las_proc\\OUTPUT_FILES\\DEM\\'
# dem_file_template = '<DATE>_dem_r<RES>m_q<QUANT>.tif'
# count_file_template = '<DATE>_dem_r<RES>m_count.tif'
#
# hs_dir_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\mb_65\\hs\\<DDI>-<DDJ>\\'
# hs_file_template = 'hs_<DDI>-<DDJ>_r<RES>_q<QUANT>.tif'

# # hs debacle
# ps - point surface
# pst - point surface thinned
# ss - surface surface
# sst - surface surface thinned

hs_in_dir_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\<DATE>\\<DATE>_las_proc\\TEMP_FILES\\15_hs\\res_<RES>\\'
hs_merged_dir_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\<DATE>\\<DATE>_las_proc\\OUTPUT_FILES\\HS\\'
hs_merged_file_template = '<DATE>_hs_r<RES>m.tif'

dhs_dir_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\mb_65\\dHS\\<DDI>-<DDJ>\\'
dhs_file_template = 'dhs_<DDI>-<DDJ>_r<RES>m.tif'

dem_in_dir_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\<DATE>\\<DATE>_las_proc\\TEMP_FILES\\12_dem\\res_<RES>\\'
dem_merged_dir_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\<DATE>\\<DATE>_las_proc\\OUTPUT_FILES\\DEM\\'
dem_merged_file_template = '<DATE>_dem_r<RES>m.tif'

dem_int_in_dir_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\<DATE>\\<DATE>_las_proc\\TEMP_FILES\\12_dem\\interpolated_res_<RES>\\'
dem_int_merged_dir_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\<DATE>\\<DATE>_las_proc\\OUTPUT_FILES\\DEM\\interpolated\\'
dem_int_merged_file_template = '<DATE>_dem_interpolated_r<RES>m.tif'

dsm_can_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\<DATE>\\<DATE>_las_proc\\OUTPUT_FILES\\CAN\\<DATE>_spike_free_dsm_can_r<RES>m.bil'

hs_bias_file_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\validation\\lidar_hs_point_samples_error.csv"
hs_bias_file_out = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\validation\\lidar_hs_point_samples_error_ceiling.csv"

hs_bc_dir_template = hs_merged_dir_template + 'bias_corrected\\'
hs_bc_file_template = hs_merged_file_template.replace('.tif', '_bias_corrected.tif')

hs_clean_dir_template = hs_merged_dir_template + 'clean\\'
hs_clean_file_template = hs_merged_file_template.replace('.tif', '_clean.tif')

swe_dir_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\<DATE>\\<DATE>_las_proc\\OUTPUT_FILES\\SWE\\'
swe_file_template = 'swe_<DATE>_r<RES>m.tif'

dswe_dir_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\mb_65\\dSWE\\<DDI>-<DDJ>\\'
dswe_file_template = 'dswe_<DDI>-<DDJ>_r<RES>m.tif'

int_dir_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\<DATE>\\<DATE>_las_proc\\OUTPUT_FILES\\DEM\\interpolated\\'
int_file_template = '<DATE>_dem_r<RES>m_q<QUANT>_interpolated_t<ITN>.tif'

chm_dir_template = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\<DATE>\\<DATE>_las_proc\\OUTPUT_FILES\\CHM\\"
chm_file_template = "<DATE>_spike_free_chm_r<RES>m.tif"

point_dens_dir_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\<DATE>\\<DATE>_las_proc\\OUTPUT_FILES\\RAS\\'
point_dens_file_template = '<DATE>_ground_point_density_r<RES>m.bil'

initial_pts_file = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\surveys\\all_ground_points_UTM11N_uid_flagged_cover.csv"
hs_uncorrected_pts_path_out = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\validation\\lidar_hs_point_samples_uncorrected.csv"
hs_uncorrected_pts_path_out_sst = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\validation\\lidar_hs_point_samples_uncorrected_sst.csv"
hs_clean_pts_path_out = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\validation\\lidar_hs_point_samples_clean.csv"
swe_pts_path_out = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\validation\\lidar_swe_point_samples.csv"
point_dens_pts_path_out = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\validation\\lidar_point_density_point_samples.csv"
dem_pts_path_out = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\validation\\lidar_dem_point_samples.csv"


def path_sub(path, dd=None, rr=None, qq=None, ddi=None, ddj=None, itn=None, mm=None, bb=None):
    if isinstance(path, str):
        # nest pure strings in list
        path = [path]

    for ii in range(0, len(path)):
        if dd is not None:
            path[ii] = path[ii].replace('<DATE>', dd)
        if rr is not None:
            path[ii] = path[ii].replace('<RES>', rr)
        if qq is not None:
            path[ii] = path[ii].replace('<QUANT>', str(qq))
        if ddi is not None:
            path[ii] = path[ii].replace('<DDI>', str(ddi))
        if ddj is not None:
            path[ii] = path[ii].replace('<DDJ>', str(ddj))
        if itn is not None:
            path[ii] = path[ii].replace('<ITN>', str(itn))

    return ''.join(path)

# merge all dems into single outputs (culled and interpolated)
for dd in (snow_on + snow_off):
    # update file paths with date
    dem_out_dir = path_sub(dem_merged_dir_template, dd=dd)
    dem_int_out_dir = path_sub(dem_int_merged_dir_template, dd=dd)

    # create DEM directory if does not exist
    if not os.path.exists(dem_out_dir):
        os.makedirs(dem_out_dir)

    if not os.path.exists(dem_int_out_dir):
        os.makedirs(dem_int_out_dir)

    for rr in resolution:
        # standard
        dem_in_dir = path_sub(dem_in_dir_template, dd=dd, rr=rr)
        dem_out_file = path_sub(dem_merged_file_template, dd=dd, rr=rr)
        rastools.raster_merge(dem_in_dir, '.bil', dem_out_dir + dem_out_file, no_data="-9999")

        # interpolated
        dem_int_in_dir = path_sub(dem_int_in_dir_template, dd=dd, rr=rr)
        dem_int_out_file = path_sub(dem_int_merged_file_template, dd=dd, rr=rr)
        rastools.raster_merge(dem_int_in_dir, '.bil', dem_int_out_dir + dem_int_out_file, no_data="-9999")

# create snow off CHM products
for dd in snow_off:
    chm_out_dir = path_sub(chm_dir_template, dd=dd)

    # create directory if does not exist
    if not os.path.exists(chm_out_dir):
        os.makedirs(chm_out_dir)

    for rr in resolution:
        dem_int_in = path_sub([dem_int_merged_dir_template, dem_int_merged_file_template], dd=dd, rr=rr)
        dsm_can_in = path_sub(dsm_can_template, dd=dd, rr=rr)
        chm_out = path_sub([chm_dir_template, chm_file_template], dd=dd, rr=rr)

        hs = rastools.raster_dif_gdal(dsm_can_in, dem_int_in, inherit_from=2, dif_out=chm_out)

# merge snow on snow depths into single output (PST)
for dd in snow_on:
    # update file paths with date
    hs_out_dir = path_sub(hs_merged_dir_template, dd=dd)

    # create DEM directory if does not exist
    if not os.path.exists(hs_out_dir):
        os.makedirs(hs_out_dir)

    for rr in resolution:
        hs_in_dir = path_sub(hs_in_dir_template, dd=dd, rr=rr)
        hs_out_file = path_sub(hs_merged_file_template, dd=dd, rr=rr)

        # calculate hs
        rastools.raster_merge(hs_in_dir, '.bil', hs_out_dir + hs_out_file, no_data="-9999")

# point hs samples
pts_file_in = initial_pts_file
for dd in snow_on:
    for rr in resolution:
        hs_in_path = path_sub(hs_merged_dir_template + hs_merged_file_template, dd=dd, rr=rr)
        colname = str(dd) + '_' + str(rr)
        rastools.csv_sample_raster(hs_in_path, pts_file_in, hs_uncorrected_pts_path_out, "xcoordUTM11", "ycoordUTM11", colname,
                                   sample_no_data_value='')
        pts_file_in = hs_uncorrected_pts_path_out


# run r script "snow_depth_bias_correction.r"

# bias-correct snow depths
hs_bias = pd.read_csv(hs_bias_file_in).loc[:, ["day", "lidar_res", "hs_mb"]]
hs_bias.loc[:, "q_999"] = np.nan
for dd in snow_on:
    # update file paths with date
    hs_in_dir = path_sub(hs_merged_dir_template, dd=dd)
    hs_bc_dir = path_sub(hs_bc_dir_template, dd=dd)

    # create DEM directory if does not exist
    if not os.path.exists(hs_bc_dir):
        os.makedirs(hs_bc_dir)

    for rr in resolution:
        hs_in_file = path_sub(hs_merged_file_template, dd=dd, rr=rr)

        # load file
        hs_bc_file = path_sub(hs_bc_file_template, dd=dd, rr=rr)

        # mean bias value
        mb = hs_bias.hs_mb[(hs_bias.day == dd) & (hs_bias.lidar_res == float(rr))]

        if len(mb) != 1:
            raise Exception("More than one (or no) match for snow depth bias, bias correction aborted.")
        mb = mb.values[0]

        # bias correct valid hs values
        ras = rastools.raster_load(hs_in_dir + hs_in_file)
        ras.data[ras.data != ras.no_data] += -mb

        # save
        rastools.raster_save(ras, hs_bc_dir + hs_bc_file)



hs_bias = pd.read_csv(hs_bias_file_in).loc[:, ["day", "lidar_res", "hs_mb"]]
hs_bias.loc[:, "ceiling_quantile"] = hs_clean_ceiling_quantile
hs_bias.loc[:, "ceiling_res"] = hs_clean_ceiling_res
hs_bias.loc[:, "ceiling_value"] = np.nan
# clean snow depths (restrict to specified range)
for dd in snow_on:
    # update file paths with date
    hs_in_dir = path_sub(hs_bc_dir_template, dd=dd)
    hs_clean_dir = path_sub(hs_clean_dir_template, dd=dd)

    # create DEM directory if does not exist
    if not os.path.exists(hs_clean_dir):
        os.makedirs(hs_clean_dir)

    # determine ceiling value
    rr = hs_clean_ceiling_res

    # load file
    hs_in_file = path_sub(hs_bc_file_template, dd=dd, rr=rr)
    ras = rastools.raster_load(hs_in_dir + hs_in_file)


    # record quantiles
    cv = np.quantile(ras.data[ras.data != ras.no_data], hs_clean_ceiling_quantile)
    hs_bias.loc[hs_bias.day == dd, "ceiling_value"] = cv

    for rr in resolution:
        hs_in_file = path_sub(hs_bc_file_template, dd=dd, rr=rr)

        # load file
        hs_clean_file = path_sub(hs_clean_file_template, dd=dd, rr=rr)

        # send negative values to zero
        ras = rastools.raster_load(hs_in_dir + hs_in_file)
        ras.data[(ras.data < 0) & (ras.data != ras.no_data)] = 0

        # send values beyond ceiling to no_data
        ras.data[ras.data > cv] = ras.no_data

        # save
        rastools.raster_save(ras, hs_clean_dir + hs_clean_file)

hs_bias.to_csv(hs_bias_file_out, index=False)

# differential snow depth products
for ii in range(0, len(snow_on)):
    ddi = snow_on[ii]
    for jj in range(ii + 1, len(snow_on)):
        ddj = snow_on[jj]
        # update file paths with dates
        dhs_dir = path_sub(dhs_dir_template, ddi=ddi, ddj=ddj)

        # create SWE directory if does not exist
        if not os.path.exists(dhs_dir):
            os.makedirs(dhs_dir)

        for rr in resolution:
            ddi_in = path_sub([hs_clean_dir_template, hs_clean_file_template], dd=ddi, rr=rr)
            ddj_in = path_sub([hs_clean_dir_template, hs_clean_file_template], dd=ddj, rr=rr)
            dhs_out = path_sub([dhs_dir_template, dhs_file_template], ddi=ddi, ddj=ddj, rr=rr)

            hs = rastools.raster_dif_gdal(ddj_in, ddi_in, inherit_from=2, dif_out=dhs_out)


# run density analysis in r

# calculate SWE products
for dd in snow_on:
    # update file paths with date
    swe_dir = path_sub(swe_dir_template, dd=dd)

    # create SWE directory if does not exist
    if not os.path.exists(swe_dir):
        os.makedirs(swe_dir)

    for rr in resolution:
        # update file paths with resolution
        hs_file = path_sub([hs_clean_dir_template, hs_clean_file_template], dd=dd, rr=rr)
        swe_file = path_sub([swe_dir_template, swe_file_template], dd=dd, rr=rr)

        # calculate swe
        ras = rastools.raster_load(hs_file)
        valid_cells = np.where(ras.data != ras.no_data)
        depth = ras.data[valid_cells]

        # juggle regression types
        if depth_regression == 'density':
            mm = depth_to_density_slope[dd]
            bb = depth_to_density_intercept[dd]
            swe = depth * (mm * depth + bb)
        elif depth_regression == 'swe':
            mm = depth_to_swe_slope[dd]
            swe = mm * depth
        elif depth_regression == 'exp':
            depth_cm = depth * 100
            swe = aa[dd] * depth_cm / 100 - (bb[dd] / 100) * (1 - np.exp(-depth_cm / cc[dd]))
        else:
            raise Exception('Invalid specification for depth_regression.')

        ras.data[valid_cells] = swe
        rastools.raster_save(ras, swe_file)


# differential SWE products
for ii in range(0, len(snow_on)):
    ddi = snow_on[ii]
    for jj in range(ii + 1, len(snow_on)):
        ddj = snow_on[jj]
        # update file paths with dates
        dswe_dir = path_sub(dswe_dir_template, ddi=ddi, ddj=ddj)

        # create SWE directory if does not exist
        if not os.path.exists(dswe_dir):
            os.makedirs(dswe_dir)

        for rr in resolution:
            ddi_in = path_sub([swe_dir_template, swe_file_template], dd=ddi, rr=rr)
            ddj_in = path_sub([swe_dir_template, swe_file_template], dd=ddj, rr=rr)
            dswe_out = path_sub([dswe_dir_template, dswe_file_template], ddi=ddi, ddj=ddj, rr=rr)

            hs = rastools.raster_dif_gdal(ddj_in, ddi_in, inherit_from=2, dif_out=dswe_out)

# spatial merge of tiled dem raster files
for dd in snow_off:
    # update file paths with date
    dem_out_dir = path_sub(dem_merged_dir_template, dd=dd)
    dem_int_out_dir = path_sub(dem_int_merged_dir_template, dd=dd)

    # create DEM directory if does not exist
    if not os.path.exists(dem_out_dir):
        os.makedirs(dem_out_dir)

    if not os.path.exists(dem_int_out_dir):
        os.makedirs(dem_int_out_dir)

    for rr in resolution:
        # standard
        dem_in_dir = path_sub(dem_in_dir_template, dd=dd, rr=rr)
        dem_out_file = path_sub(dem_merged_file_template, dd=dd, rr=rr)
        rastools.raster_merge(dem_in_dir, '.bil', dem_out_dir + dem_out_file, no_data="-9999")

        # interpolated
        dem_int_in_dir = path_sub(dem_int_in_dir_template, dd=dd, rr=rr)
        dem_int_out_file = path_sub(dem_int_merged_file_template, dd=dd, rr=rr)
        rastools.raster_merge(dem_int_in_dir, '.bil', dem_int_out_dir + dem_int_out_file, no_data="-9999")


# point samples of hs
pts_file_in = initial_pts_file
for dd in snow_on:
    for rr in resolution:
        hs_clean_path = path_sub(hs_clean_dir_template + hs_clean_file_template, dd=dd, rr=rr)
        colname = str(dd) + '_' + str(rr)
        rastools.csv_sample_raster(hs_clean_path, pts_file_in, hs_clean_pts_path_out, "xcoordUTM11", "ycoordUTM11", colname,
                                   sample_no_data_value='')
        pts_file_in = hs_clean_pts_path_out

# point samples of swe
pts_file_in = initial_pts_file
for dd in snow_on:
    for rr in resolution:
        swe_path = path_sub(swe_dir_template + swe_file_template, dd=dd, rr=rr)
        colname = str(dd) + '_' + str(rr)
        rastools.csv_sample_raster(swe_path, pts_file_in, swe_pts_path_out, "xcoordUTM11", "ycoordUTM11", colname, sample_no_data_value='')
        pts_file_in = swe_pts_path_out


# point samples of point density
pts_file_in = initial_pts_file
for dd in snow_on + snow_off:
    for rr in [".10", ".25"]:
        point_dens_path = path_sub(point_dens_dir_template + point_dens_file_template, dd=dd, rr=rr)
        colname = str(dd) + '_' + str(rr)

        rastools.csv_sample_raster(point_dens_path, pts_file_in, point_dens_pts_path_out, "xcoordUTM11", "ycoordUTM11", colname, sample_no_data_value='')
        pts_file_in = point_dens_pts_path_out

        print(rr)

# point samples of interpolated dem
pts_file_in = initial_pts_file
for dd in (snow_on + snow_off):
    for rr in resolution:
        dem_path = path_sub(dem_merged_dir_template + dem_merged_file_template, dd=dd, rr=rr)
        colname = str(dd) + '_' + str(rr)

        rastools.csv_sample_raster(dem_path, pts_file_in, dem_pts_path_out, "xcoordUTM11", "ycoordUTM11", colname, sample_no_data_value='')
        pts_file_in = dem_pts_path_out

        print(rr)

# point sample HS products to merge with snow surveys

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
#
# plt.plot(rank, values)
#
#
#
# nn = len(rank)
# v1 = np.convolve(values, np.ones(int(nn/100000)), 'valid')/int(nn/100000)
# r1 = np.arange(0, len(v1))/len(v1)
# nn = len(r1)
#
# plt.plot(r1, v1)
#
# v2 = v1[1:nn] - v1[0:nn-1]
# r2 = r1[0:nn-1]
# plt.plot(r2, v2)