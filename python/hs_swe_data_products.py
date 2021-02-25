import rastools
import numpy as np
import pandas as pd
import os

snow_on_ass = {}
snow_on_ass["alin"] = ["19_045", "19_050", "19_052", "19_107", "19_123"]
snow_on_ass["ahpl"] = ["19_045", "19_050", "19_052"]
snow_on = ["19_045", "19_050", "19_052", "19_107", "19_123"]
# snow_on = ["19_045", "19_050", "19_052"]
snow_off = ["19_149"]
all_dates = snow_on + snow_off

resolution = [".05", ".10", ".25", "1.00"]

interpolation_lengths = ["1", "2", "3"]

dens_ass = {}
#
# coefficients for snow depth [cm], swe [mm]
# all veg, each day, linear depth-density
depth_to_density_intercept = dict(zip(snow_on, [109.1403, 110.2249, 72.5015, 224.6406, 223.5683]))
depth_to_density_slope = dict(zip(snow_on, np.array([1.2717, 1.2212, 1.5346, 1.7833, 1.2072])))
dens_ass["alin"] = (depth_to_density_intercept, depth_to_density_slope)

# # forest only, each day, linear depth-density
# depth_to_density_intercept = dict(zip(snow_on, [147.5136, 102.460, 3.303, 249.1015, 293.10207]))
# depth_to_density_slope = dict(zip(snow_on, np.array([1.3616, 1.486, 4.054, 0.3966, -0.03987])))
# dens_ass["flin"] = (depth_to_density_intercept, depth_to_density_slope)
#
# # all, each day, constant density
# depth_to_density_intercept = dict(zip(snow_on, np.array([191.534, 189.129, 176.066, 297.336, 298.589])))
# depth_to_density_slope = dict(zip(snow_on, np.array([0, 0, 0, 0, 0])))
# dens_ass["acon"] = (depth_to_density_intercept, depth_to_density_slope)
#
# # forest only, each day, constant density
# depth_to_density_intercept = dict(zip(snow_on, np.array([199.321, 158.56, 134.48, 263.22, 291.14])))
# depth_to_density_slope = dict(zip(snow_on, np.array([0, 0, 0, 0, 0])))
# dens_ass["fcon"] = (depth_to_density_intercept, depth_to_density_slope)
#
# hedstrom pomeroy intercept linear
depth_to_density_intercept = dict(zip(snow_on, np.array([89.26, 85.39, 72.05])))
depth_to_density_slope = dict(zip(snow_on, np.array([1.59, 1.6336, 1.5420])))
dens_ass["ahpl"] = (depth_to_density_intercept, depth_to_density_slope)

# adjacent day combined linear density models
# ajli_dates = np.array(["19_045", "19_050", "19_052"])
# ajli_intercept = np.array([148.7251, 97.5015])
# ajli_slope = np.array([0.6864, 1.2815])


dem_in_dir_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\<DATE>\\<DATE>_las_proc\\TEMP_FILES\\12_dem\\res_<RES>\\'
dem_merged_dir_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\<DATE>\\<DATE>_las_proc\\OUTPUT_FILES\\DEM\\'
dem_merged_file_template = '<DATE>_dem_r<RES>m.tif'

dem_int_in_dir_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\<DATE>\\<DATE>_las_proc\\TEMP_FILES\\12_dem\\interpolated_res_<RES>\\'
dem_int_merged_dir_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\<DATE>\\<DATE>_las_proc\\OUTPUT_FILES\\DEM\\interpolated\\'
dem_int_merged_file_template = '<DATE>_dem_interpolated_r<RES>m.tif'

dsm_can_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\<DATE>\\<DATE>_las_proc\\OUTPUT_FILES\\CAN\\<DATE>_spike_free_dsm_can_r<RES>m.bil'
chm_dir_template = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\<DATE>\\<DATE>_las_proc\\OUTPUT_FILES\\CHM\\"
chm_file_template = "<DATE>_spike_free_chm_r<RES>m.tif"

hs_in_dir_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\<DATE>\\<DATE>_las_proc\\TEMP_FILES\\15_hs\\interp_<INTLEN>x\\res_<RES>\\'
hs_merged_dir_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\<DATE>\\<DATE>_las_proc\\OUTPUT_FILES\\HS\\interp_<INTLEN>x\\'
hs_merged_file_template = '<DATE>_hs_r<RES>m_interp<INTLEN>x.tif'

hs_bias_file_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\validation\\lidar_hs_point_samples_error.csv"
hs_bias_file_out = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\validation\\lidar_hs_point_samples_error_ceiling.csv"

hs_bc_dir_template = hs_merged_dir_template + 'bias_corrected\\'
hs_bc_file_template = hs_merged_file_template.replace('.tif', '_bias_corrected.tif')

hs_clean_dir_template = hs_merged_dir_template + 'clean\\'
hs_clean_file_template = hs_merged_file_template.replace('.tif', '_clean.tif')

hs_eroded_dir_template = hs_merged_dir_template + 'eroded\\'
hs_eroded_file_template = hs_merged_file_template.replace('.tif', '_eroded.tif')

dhs_dir_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\mb_65\\dHS\\interp_<INTLEN>x\\<DDI>-<DDJ>\\'
dhs_file_template = 'dhs_<DDI>-<DDJ>_r<RES>m_interp<INTLEN>x.tif'

swe_dir_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\<DATE>\\<DATE>_las_proc\\OUTPUT_FILES\\SWE\\<ASS>\\interp_<INTLEN>x\\'
swe_file_template = 'swe_<ASS>_<DATE>_r<RES>m_interp<INTLEN>x.tif'

swe_masked_dir_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\<DATE>\\<DATE>_las_proc\\OUTPUT_FILES\\SWE\\<ASS>\\interp_<INTLEN>x\\masked\\'
swe_masked_file_template = 'swe_<ASS>_<DATE>_r<RES>m_interp<INTLEN>x_masked.tif'

dswe_dir_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\mb_65\\dSWE\\<ASS>\\interp_<INTLEN>x\\<DDI>-<DDJ>\\'
dswe_file_template = 'dswe_<ASS>_<DDI>-<DDJ>_r<RES>m_interp<INTLEN>x.tif'

dswe_masked_dir_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\mb_65\\dSWE\\<ASS>\\interp_<INTLEN>x\\<DDI>-<DDJ>\\masked\\'
dswe_masked_file_template = 'dswe_<ASS>_<DDI>-<DDJ>_r<RES>m_interp<INTLEN>x_masked.tif'

# THIS MAY BE TRASH...
# int_dir_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\<DATE>\\<DATE>_las_proc\\OUTPUT_FILES\\DEM\\interpolated\\'
# int_file_template = '<DATE>_dem_r<RES>m_q<QUANT>_interpolated_t<ITN>.tif'
#

point_dens_dir_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\<DATE>\\<DATE>_las_proc\\OUTPUT_FILES\\RAS\\'
point_dens_file_template = '<DATE>_ground_point_density_r<RES>m.bil'

initial_pts_file = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\surveys\\all_ground_points_UTM11N_uid_flagged_cover.csv"
hs_uncorrected_pts_path_out = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\validation\\lidar_hs_point_samples_uncorrected.csv"
hs_uncorrected_pts_path_out_sst = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\validation\\lidar_hs_point_samples_uncorrected_sst.csv"
hs_clean_pts_path_out = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\validation\\lidar_hs_point_samples_clean.csv"
swe_pts_path_out = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\validation\\lidar_swe_point_samples.csv"
point_dens_pts_path_out = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\validation\\lidar_point_density_point_samples.csv"
dem_pts_path_out = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\validation\\lidar_dem_point_samples.csv"

# mask polygons
snow_mask = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\site_library\\snow_depth_mask.shp"
trail_mask = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\site_library\\trampled_snow_mask_dissolved.shp"


def path_sub(path, dd=None, rr=None, qq=None, ddi=None, ddj=None, itn=None, ass=None, intlen=None):
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
        if ass is not None:
            path[ii] = path[ii].replace('<ASS>', str(ass))
        if intlen is not None:
            path[ii] = path[ii].replace('<INTLEN>', str(intlen))

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
    for intlen in interpolation_lengths:
        # update file paths with date
        hs_out_dir = path_sub(hs_merged_dir_template, dd=dd, intlen=intlen)

        # create DEM directory if does not exist
        if not os.path.exists(hs_out_dir):
            os.makedirs(hs_out_dir)

        for rr in resolution:
            hs_in_dir = path_sub(hs_in_dir_template, dd=dd, rr=rr, intlen=intlen)
            hs_out_file = path_sub(hs_merged_file_template, dd=dd, rr=rr, intlen=intlen)

            # calculate hs
            rastools.raster_merge(hs_in_dir, '.bil', hs_out_dir + hs_out_file, no_data="-9999")


# point hs samples
pts_file_in = initial_pts_file
for dd in snow_on:
    for intlen in interpolation_lengths:
        for rr in resolution:
            hs_in_path = path_sub(hs_merged_dir_template + hs_merged_file_template, dd=dd, rr=rr, intlen=intlen)
            colname = str(dd) + '_' + str(rr) + '_' + str(intlen)
            rastools.csv_sample_raster(hs_in_path, pts_file_in, hs_uncorrected_pts_path_out, "xcoordUTM11", "ycoordUTM11", colname, sample_no_data_value='')
            pts_file_in = hs_uncorrected_pts_path_out

# run r script "snow_depth_bias_correction.r"

# bias-correct snow depths
hs_bias = pd.read_csv(hs_bias_file_in).loc[:, ["day", "lidar_res", "interp_len", "hs_mb"]]
hs_bias.loc[:, "q_999"] = np.nan
hs_bias_cor_res = 0.05
hs_bias_cor_intlen = 2
for dd in snow_on:
    for intlen in interpolation_lengths:
        # update file paths with date
        hs_in_dir = path_sub(hs_merged_dir_template, dd=dd, intlen=intlen)
        hs_bc_dir = path_sub(hs_bc_dir_template, dd=dd, intlen=intlen)

        # create DEM directory if does not exist
        if not os.path.exists(hs_bc_dir):
            os.makedirs(hs_bc_dir)

        # read mean bias value
        mb = hs_bias.hs_mb[(hs_bias.day == dd) &
                           (hs_bias.lidar_res == hs_bias_cor_res) &
                           (hs_bias.interp_len == hs_bias_cor_intlen)]
        if len(mb) != 1:
            raise Exception("More than one (or no) match for snow depth bias, bias correction aborted.")
        mb = mb.values[0]
        # mb = 0  # no bias correction!

        for rr in resolution:
            hs_in_file = path_sub(hs_merged_file_template, dd=dd, rr=rr, intlen=intlen)

            # load file
            hs_bc_file = path_sub(hs_bc_file_template, dd=dd, rr=rr, intlen=intlen)

            # bias correct valid hs values
            ras = rastools.raster_load(hs_in_dir + hs_in_file)
            ras.data[ras.data != ras.no_data] += -mb

            # save
            rastools.raster_save(ras, hs_bc_dir + hs_bc_file)


hs_clean_ceiling_res = '.05'
hs_clean_ceiling_intlen = '2'  # avoid intlen 1 with res .05
hs_clean_ceiling_quantile = 0.999  # determined visually...
hs_bias = pd.read_csv(hs_bias_file_in).loc[:, ["day", "lidar_res", "interp_len", "hs_mb"]]
hs_bias.loc[:, "ceiling_quantile"] = hs_clean_ceiling_quantile
hs_bias.loc[:, "ceiling_res"] = hs_clean_ceiling_res
hs_bias.loc[:, "ceiling_value"] = np.nan
# clean snow depths (restrict to specified range)
for dd in snow_on:
    for intlen in interpolation_lengths:
        # update file paths with date
        hs_in_dir = path_sub(hs_bc_dir_template, dd=dd, intlen=intlen)
        hs_clean_dir = path_sub(hs_clean_dir_template, dd=dd, intlen=intlen)

        # create DEM directory if does not exist
        if not os.path.exists(hs_clean_dir):
            os.makedirs(hs_clean_dir)

        # load file
        hs_ceil_path = path_sub(hs_bc_dir_template + hs_bc_file_template, dd=dd, rr=hs_clean_ceiling_res, intlen=hs_clean_ceiling_intlen)
        ras = rastools.raster_load(hs_ceil_path)

        # record quantiles
        cv = np.quantile(ras.data[ras.data != ras.no_data], hs_clean_ceiling_quantile)
        hs_bias.loc[hs_bias.day == dd, "ceiling_value"] = cv

        for rr in resolution:
            hs_in_file = path_sub(hs_bc_file_template, dd=dd, rr=rr, intlen=intlen)

            # load file
            hs_clean_file = path_sub(hs_clean_file_template, dd=dd, rr=rr, intlen=intlen)

            # send negative values to zero
            ras = rastools.raster_load(hs_in_dir + hs_in_file)
            ras.data[(ras.data < 0) & (ras.data != ras.no_data)] = 0

            # send values beyond ceiling to no_data
            ras.data[ras.data > cv] = ras.no_data

            # save
            rastools.raster_save(ras, hs_clean_dir + hs_clean_file)

hs_bias.to_csv(hs_bias_file_out, index=False)

# from scipy import ndimage
#
# fsize = 11
# yy, xx = np.mgrid[0:fsize, 0:fsize]
# cc = (fsize-1)/2
# dist = np.sqrt((xx - cc) ** 2 + (yy - cc) ** 2)
# footprint = (dist <= cc)
#
# # snow depth erosion
# for dd in snow_on:
#     for intlen in interpolation_lengths:
#         # update file paths with date
#         hs_in_dir = path_sub(hs_clean_dir_template, dd=dd, intlen=intlen)
#         hs_eroded_dir = path_sub(hs_eroded_dir_template, dd=dd, intlen=intlen)
#
#         # create DEM directory if does not exist
#         if not os.path.exists(hs_eroded_dir):
#             os.makedirs(hs_eroded_dir)
#
#
#         for rr in resolution:
#             # in file
#             hs_in_file = path_sub(hs_clean_file_template, dd=dd, rr=rr, intlen=intlen)
#
#             # out file
#             hs_eroded_file = path_sub(hs_eroded_file_template, dd=dd, rr=rr, intlen=intlen)
#
#             # load data
#             ras = rastools.raster_load(hs_in_dir + hs_in_file)
#
#             dat = ras.data.copy()
#             na_dat = (dat == ras.no_data)
#
#             dat[na_dat] = np.nan
#
#             err = ndimage.grey_erosion(dat, footprint=footprint)
#
#             ras.data = err
#             ras.data[na_dat] = ras.no_data
#
#             # save
#             rastools.raster_save(ras, hs_eroded_dir + hs_eroded_file)

# differential snow depth products
for ii in range(0, len(snow_on) - 1):
    ddi = snow_on[ii]
    ddj = snow_on[ii + 1]
    for intlen in interpolation_lengths:
        # update file paths with dates
        dhs_dir = path_sub(dhs_dir_template, ddi=ddi, ddj=ddj, intlen=intlen)

        # create SWE directory if does not exist
        if not os.path.exists(dhs_dir):
            os.makedirs(dhs_dir)

        for rr in resolution:
            ddi_in = path_sub([hs_clean_dir_template, hs_clean_file_template], dd=ddi, rr=rr, intlen=intlen)
            ddj_in = path_sub([hs_clean_dir_template, hs_clean_file_template], dd=ddj, rr=rr, intlen=intlen)
            dhs_out = path_sub([dhs_dir_template, dhs_file_template], ddi=ddi, ddj=ddj, rr=rr, intlen=intlen)

            hs = rastools.raster_dif_gdal(ddj_in, ddi_in, inherit_from=2, dif_out=dhs_out)


# run density analysis in r
#
# calculate SWE products
for ass in dens_ass.keys():
    print(ass)
    for intlen in interpolation_lengths:
        for dd in snow_on_ass[ass]:
            # update file paths with date
            swe_dir = path_sub(swe_dir_template, dd=dd, ass=ass, intlen=intlen)
            swe_masked_dir = path_sub(swe_masked_dir_template, dd=dd, ass=ass, intlen=intlen)

            # create SWE directory if does not exist
            if not os.path.exists(swe_dir):
                os.makedirs(swe_dir)
            if not os.path.exists(swe_masked_dir):
                os.makedirs(swe_masked_dir)

            for rr in resolution:
                # update file paths with resolution
                hs_file = path_sub([hs_clean_dir_template, hs_clean_file_template], dd=dd, rr=rr, intlen=intlen)
                swe_file = path_sub([swe_dir_template, swe_file_template], dd=dd, rr=rr, ass=ass, intlen=intlen)
                swe_masked_file = path_sub([swe_masked_dir_template, swe_masked_file_template], dd=dd, rr=rr, ass=ass, intlen=intlen)

                # calculate swe
                ras = rastools.raster_load(hs_file)
                valid_cells = np.where(ras.data != ras.no_data)
                depth = ras.data[valid_cells]

                # calculate swe from depth density regression
                mm = dens_ass[ass][1][dd]
                bb = dens_ass[ass][0][dd]
                swe = depth * (mm * depth * 100 + bb)

                ras.data[valid_cells] = swe
                rastools.raster_save(ras, swe_file)

                # mask
                ras = rastools.raster_load(swe_file)
                rastools.raster_save(ras, swe_masked_file)

                rastools.raster_burn(swe_masked_file, snow_mask, ras.no_data)
                rastools.raster_burn(swe_masked_file, trail_mask, ras.no_data)


# differential SWE products
for ass in dens_ass.keys():
    print(ass)
    for intlen in interpolation_lengths:
        for ii in range(0, len(snow_on_ass[ass]) - 1):
            ddi = snow_on[ii]
            ddj = snow_on[ii + 1]

            # update file paths with dates
            dswe_dir = path_sub(dswe_dir_template, ddi=ddi, ddj=ddj, ass=ass, intlen=intlen)
            dswe_masked_dir = path_sub(dswe_masked_dir_template, ddi=ddi, ddj=ddj, ass=ass, intlen=intlen)

            # create SWE directory if does not exist
            if not os.path.exists(dswe_dir):
                os.makedirs(dswe_dir)
            if not os.path.exists(dswe_masked_dir):
                os.makedirs(dswe_masked_dir)

            for rr in resolution:
                ddi_in = path_sub([swe_dir_template, swe_file_template], dd=ddi, rr=rr, ass=ass, intlen=intlen)
                ddj_in = path_sub([swe_dir_template, swe_file_template], dd=ddj, rr=rr, ass=ass, intlen=intlen)
                dswe_file = path_sub([dswe_dir_template, dswe_file_template], ddi=ddi, ddj=ddj, rr=rr, ass=ass, intlen=intlen)
                dswe_masked_file = path_sub([dswe_masked_dir_template, dswe_masked_file_template], ddi=ddi, ddj=ddj, rr=rr, ass=ass, intlen=intlen)

                dswe_ras = rastools.raster_dif_gdal(ddj_in, ddi_in, inherit_from=2, dif_out=dswe_file)

                # mask
                ras = rastools.raster_load(dswe_file)
                rastools.raster_save(ras, dswe_masked_file)

                rastools.raster_burn(dswe_masked_file, snow_mask, ras.no_data)
                rastools.raster_burn(dswe_masked_file, trail_mask, ras.no_data)


#
# # adjacent combined linear (ajli) density SWE products
# ii = 0
# for dd in ajli_dates[0:2]:
#     for intlen in interpolation_lengths:
#         # update file paths with date
#         swe_dir = path_sub(swe_dir_template, dd=dd, ass="ajli_1", intlen=intlen)
#
#         # create SWE directory if does not exist
#         if not os.path.exists(swe_dir):
#             os.makedirs(swe_dir)
#
#         for rr in resolution:
#             # update file paths with resolution
#             hs_file = path_sub([hs_clean_dir_template, hs_clean_file_template], dd=dd, rr=rr, intlen=intlen)
#             swe_file = path_sub([swe_dir_template, swe_file_template], dd=dd, rr=rr, ass="ajli_1", intlen=intlen)
#
#             # calculate swe
#             ras = rastools.raster_load(hs_file)
#             valid_cells = np.where(ras.data != ras.no_data)
#             depth = ras.data[valid_cells]
#
#             # calculate swe from depth density regression
#             mm = ajli_slope[ii]
#             bb = ajli_intercept[ii]
#             swe = depth * (mm * depth * 100 + bb)
#
#             print(str(mm))
#
#             ras.data[valid_cells] = swe
#             rastools.raster_save(ras, swe_file)
#     ii += 1
#
# ii = 0
# for dd in ajli_dates[1:3]:
#     for intlen in interpolation_lengths:
#         # update file paths with date
#         swe_dir = path_sub(swe_dir_template, dd=dd, ass="ajli_2", intlen=intlen)
#
#         # create SWE directory if does not exist
#         if not os.path.exists(swe_dir):
#             os.makedirs(swe_dir)
#
#         for rr in resolution:
#             # update file paths with resolution
#             hs_file = path_sub([hs_clean_dir_template, hs_clean_file_template], dd=dd, rr=rr, intlen=intlen)
#             swe_file = path_sub([swe_dir_template, swe_file_template], dd=dd, rr=rr, ass="ajli_2", intlen=intlen)
#
#             # calculate swe
#             ras = rastools.raster_load(hs_file)
#             valid_cells = np.where(ras.data != ras.no_data)
#             depth = ras.data[valid_cells]
#
#             # calculate swe from depth density regression
#             mm = ajli_slope[ii]
#             bb = ajli_intercept[ii]
#             swe = depth * (mm * depth * 100 + bb)
#
#             print(str(mm))
#
#             ras.data[valid_cells] = swe
#             rastools.raster_save(ras, swe_file)
#     ii += 1
#
# # differential SWE products
# for ii in range(0, len(ajli_dates) - 1):
#     ddi = ajli_dates[ii]
#     ddj = ajli_dates[ii + 1]
#     for intlen in interpolation_lengths:
#
#         # update file paths with dates
#         dswe_dir = path_sub(dswe_dir_template, ddi=ddi, ddj=ddj, ass="ajli", intlen=intlen)
#
#         # create SWE directory if does not exist
#         if not os.path.exists(dswe_dir):
#             os.makedirs(dswe_dir)
#
#         for rr in resolution:
#             ddi_in = path_sub([swe_dir_template, swe_file_template], dd=ddi, rr=rr, ass="ajli_1", intlen=intlen)
#             ddj_in = path_sub([swe_dir_template, swe_file_template], dd=ddj, rr=rr, ass="ajli_2", intlen=intlen)
#             dswe_out = path_sub([dswe_dir_template, dswe_file_template], ddi=ddi, ddj=ddj, rr=rr, ass="ajli", intlen=intlen)
#
#             hs = rastools.raster_dif_gdal(ddj_in, ddi_in, inherit_from=2, dif_out=dswe_out)

# masked differential SWE
snow_mask = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\site_library\\snow_depth_mask.shp"
trail_mask = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\site_library\\trampled_snow_mask_dissolved.shp"
for ass in dens_ass.keys():
    print(ass)
    for intlen in interpolation_lengths:
        for ii in range(0, len(snow_on_ass[ass]) - 1):
            ddi = snow_on[ii]
            ddj = snow_on[ii + 1]


            # update file paths with dates
            dswe_masked_dir = path_sub(dswe_masked_dir_template, ddi=ddi, ddj=ddj, ass=ass, intlen=intlen)

            # create SWE directory if does not exist
            if not os.path.exists(dswe_masked_dir):
                os.makedirs(dswe_masked_dir)

            for rr in resolution:
                old_file = path_sub(dswe_dir_template + dswe_file_template, ddi=ddi, ddj=ddj, rr=rr, ass=ass,
                                    intlen=intlen)
                new_file = path_sub(dswe_masked_dir_template + dswe_masked_file_template, ddi=ddi, ddj=ddj, rr=rr,
                                    ass=ass, intlen=intlen)

                ras = rastools.raster_load(old_file)
                rastools.raster_save(ras, new_file)

                rastools.raster_burn(new_file, snow_mask, ras.no_data)
                rastools.raster_burn(new_file, trail_mask, ras.no_data)

# for ii in range(0, len(ajli_dates) - 1):
#     ddi = ajli_dates[ii]
#     ddj = ajli_dates[ii + 1]
#     for intlen in interpolation_lengths:
#
#         # update file paths with dates
#         dswe_masked_dir = path_sub(dswe_masked_dir_template, ddi=ddi, ddj=ddj, ass="ajli", intlen=intlen)
#
#         # create SWE directory if does not exist
#         if not os.path.exists(dswe_masked_dir):
#             os.makedirs(dswe_masked_dir)
#
#         for rr in resolution:
#             old_file = path_sub(dswe_dir_template + dswe_file_template, ddi=ddi, ddj=ddj, rr=rr, ass="ajli", intlen=intlen)
#             new_file = path_sub(dswe_masked_dir_template + dswe_masked_file_template, ddi=ddi, ddj=ddj, rr=rr, ass="ajli", intlen=intlen)
#
#             ras = rastools.raster_load(old_file)
#             rastools.raster_save(ras, new_file)
#
#             rastools.raster_burn(new_file, snow_mask, ras.no_data)
#             rastools.raster_burn(new_file, trail_mask, ras.no_data)



# point samples of hs
pts_file_in = initial_pts_file
for dd in snow_on:
    for intlen in interpolation_lengths:
        for rr in resolution:
            try:
                hs_clean_path = path_sub(hs_clean_dir_template + hs_clean_file_template, dd=dd, rr=rr, intlen=intlen)
                colname = str(dd) + '_' + str(rr) + '_' + str(intlen)
                rastools.csv_sample_raster(hs_clean_path, pts_file_in, hs_clean_pts_path_out, "xcoordUTM11",
                                           "ycoordUTM11", colname, sample_no_data_value='')
                pts_file_in = hs_clean_pts_path_out

            except AttributeError:
                print('File does not exist')

# # point samples of swe
# pts_file_in = initial_pts_file
# for dd in snow_on:
#     for ass in ["ajli_1", "ajli_2"]:
#         for intlen in interpolation_lengths:
#             for rr in resolution:
#                 try:
#                     swe_path = path_sub(swe_dir_template + swe_file_template, dd=dd, rr=rr, ass=ass, intlen=intlen)
#                     colname = str(dd) + '_' + str(rr) + '_' + str(ass) + '_' + str(intlen)
#                     rastools.csv_sample_raster(swe_path, pts_file_in, swe_pts_path_out, "xcoordUTM11", "ycoordUTM11", colname, sample_no_data_value='')
#                     pts_file_in = swe_pts_path_out
#                 except AttributeError:
#                     print('File does not exist')

# point samples of swe
pts_file_in = initial_pts_file
for ass in dens_ass.keys():
    for dd in snow_on_ass[ass]:
        for intlen in interpolation_lengths:
            for rr in resolution:
                try:
                    swe_path = path_sub(swe_dir_template + swe_file_template, dd=dd, rr=rr, ass=ass, intlen=intlen)
                    colname = str(dd) + '_' + str(rr) + '_' + str(ass) + '_' + str(intlen)
                    rastools.csv_sample_raster(swe_path, pts_file_in, swe_pts_path_out, "xcoordUTM11", "ycoordUTM11", colname, sample_no_data_value='')
                    pts_file_in = swe_pts_path_out
                except AttributeError:
                    print('File does not exist')


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