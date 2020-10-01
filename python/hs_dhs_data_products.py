import rastools
import laslib
import numpy as np
import os

snow_on = ["19_045", "19_050", "19_052", "19_107", "19_123"]
snow_off = ["19_149"]
all_dates = snow_on + snow_off

resolution = [".04", ".10", ".25", ".50", "1.00"]

depth_to_density_intercept = dict(zip(snow_on, [0, 0, 0, 0, 0]))
depth_to_density_slope = dict(zip(snow_on, 100*np.array([2.695, 2.7394, 3.0604, 3.1913, 2.5946])))

depth_to_swe_slope = dict(zip(snow_on, 100*np.array([2.695, 2.7394, 3.0604, 3.1913, 2.5946])))

depth_regression = 'swe'

dem_quantile = .25
interpolation_threshold = 0

las_in_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\<DATE>\\<DATE>_las_proc\\OUTPUT_FILES\\LAS\\<DATE>_las_proc_classified_merged.las'
dem_ref_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\raster_templates\\hs_19_045_res_<RES>m.tif'
dem_dir_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\<DATE>\\<DATE>_las_proc\\OUTPUT_FILES\\DEM\\'
dem_file_template = '<DATE>_dem_r<RES>m_q<QUANT>.tif'
count_file_template = '<DATE>_dem_r<RES>m_count.tif'

hs_dir_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\hs\\<DDI>-<DDJ>\\'
hs_file_template = 'hs_<DDI>-<DDJ>_r<RES>_q<QUANT>.tif'

swe_dir_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\SWE\\<DATE>\\'
swe_file_template = 'swe_<DATE>_r<RES>m_q<QUANT>.tif'

dswe_dir_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\dSWE\\<DDI>-<DDJ>\\'
dswe_file_template = 'dswe_<DDI>-<DDJ>_r<RES>m_q<QUANT>.tif'

int_dir_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\<DATE>\\<DATE>_las_proc\\OUTPUT_FILES\\DEM\\interpolated\\'
int_file_template = '<DATE>_dem_r<RES>m_q<QUANT>_interpolated_t<ITN>.tif'

chm_dir_template = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\<DATE>\\<DATE>_las_proc\\OUTPUT_FILES\\CHM\\"
chm_raw_in_template = "<DATE>_spike_free_chm_r<RES>m.bil"
chm_filled_out_template = "<DATE>_spike_free_chm_r<RES>m_filled.tif"

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


# create dems at all resolutions using quantile method
for dd in all_dates:
    # update file paths with date
    las_in = path_sub(las_in_template, dd)
    dem_dir = path_sub(dem_dir_template, dd)

    # create DEM directory if does not exist
    if not os.path.exists(dem_dir):
        os.makedirs(dem_dir)

    for rr in resolution:
        # update file paths with resolution
        ras_template = path_sub(dem_ref_template, rr=rr)
        dem_file = path_sub([dem_dir_template, dem_file_template], dd=dd, rr=rr, qq=dem_quantile)
        count_file = path_sub([dem_dir_template, count_file_template], dd=dd, rr=rr)

        # calculate dem
        stat_q, stat_n = laslib.las_quantile_dem(las_in, ras_template, dem_quantile, q_out=dem_file, n_out=count_file)

# differential dem products (snow depth/hs)
for ddi in snow_on:
    for ddj in snow_off:

        # create HS directory if does not exist
        hs_dir = path_sub(hs_dir_template, ddi=ddi, ddj=ddj)
        if not os.path.exists(hs_dir):
            os.makedirs(hs_dir)

        for rr in resolution:
            ddi_in = path_sub([dem_dir_template, dem_file_template], dd=ddi, rr=rr, qq=dem_quantile)
            ddj_in = path_sub([dem_dir_template, dem_file_template], dd=ddj, rr=rr, qq=dem_quantile)
            hs_out = path_sub([hs_dir_template, hs_file_template], ddi=ddi, ddj=ddj, rr=rr, qq=dem_quantile)

            hs = rastools.raster_dif(ddi_in, ddj_in, inherit_from=2, dif_out=hs_out)

# SWE products
for ddi in snow_on:
    for ddj in snow_off:
        # update file paths with date
        swe_dir = path_sub(swe_dir_template, ddi)

        # create SWE directory if does not exist
        if not os.path.exists(swe_dir):
            os.makedirs(swe_dir)



        for rr in resolution:
            # update file paths with resolution
            hs_file = path_sub([hs_dir_template, hs_file_template], ddi=ddi, ddj=ddj, rr=rr, qq=dem_quantile)
            swe_file = path_sub([swe_dir_template, swe_file_template], dd=ddi, rr=rr, qq=dem_quantile)

            # calculate swe
            ras = rastools.raster_load(hs_file)
            valid_cells = np.where(ras.data != ras.no_data)
            depth = ras.data[valid_cells]

            # juggle regression types
            if depth_regression == 'density':
                mm = depth_to_density_slope[ddi]
                bb = depth_to_density_intercept[ddi]
                swe = depth * (mm * depth + bb)
            elif depth_regression == 'swe':
                mm = depth_to_swe_slope[ddi]
                swe = mm * depth
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
            ddi_in = path_sub([swe_dir_template, swe_file_template], dd=ddi, rr=rr, qq=dem_quantile)
            ddj_in = path_sub([swe_dir_template, swe_file_template], dd=ddj, rr=rr, qq=dem_quantile)
            dswe_out = path_sub([dswe_dir_template, dswe_file_template], ddi=ddi, ddj=ddj, rr=rr, qq=dem_quantile)

            hs = rastools.raster_dif(ddj_in, ddi_in, inherit_from=2, dif_out=dswe_out)

# interpolated products
for dd in snow_off:
    print(dd, end='')
    # update file paths with date
    int_dir = path_sub(int_dir_template, dd)

    # create DEM directory if does not exist
    if not os.path.exists(int_dir):
        os.makedirs(int_dir)

    for rr in resolution:
        ras_template = path_sub(dem_ref_template, rr=rr)
        dem_file = path_sub([dem_dir_template, dem_file_template], dd=dd, rr=rr, qq=dem_quantile)
        count_file = path_sub([dem_dir_template, count_file_template], dd=dd, rr=rr)
        int_file = path_sub([int_dir_template, int_file_template], dd=dd, rr=rr, qq=dem_quantile, itn=interpolation_threshold)

        # interpolate dem
        rastools.delauney_fill(dem_file, int_file, ras_template, n_count=count_file, n_threshold=interpolation_threshold)
        print(' -- ' + rr, end='')
    print('\n')

# fill chm with zeros where dem in not nan
# only for dates and resolutions where chm exists
for dd in all_dates:
    for rr in resolution:
        # update file paths with resolution
        chm_in = path_sub([chm_dir_template, chm_raw_in_template], dd=dd, rr=rr)
        if os.path.exists(chm_in):
            count_file = path_sub([dem_dir_template, count_file_template], dd=dd, rr=rr)
            chm_out = path_sub([chm_dir_template, chm_filled_out_template], dd=dd, rr=rr)

            # fill in chm nan values with 0 where dem count > 0

            data = rastools.pd_sample_raster(None, chm_in, 'chm', include_nans=True)
            data = rastools.pd_sample_raster(data, count_file, 'n_count')

            ground = np.isnan(data.chm) & (data.n_count > 0)

            ground_index = (np.array(data.x_index[ground]), np.array(data.y_index[ground]))

            chm = rastools.raster_load(chm_in)
            chm.data[ground_index] = 0
            rastools.raster_save(chm, chm_out)


chm_dir_template = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\<DATE>\\<DATE>_las_proc\\OUTPUT_FILES\\CHM\\""
chm_file_raw_template = "<DATE>_spike_free_chm_r<RES>m.bil"
chm_file_filled_template = "<DATE>_spike_free_chm_r<RES>m_filled.bil"

# point sample HS products to merge with snow surveys
initial_pts_file = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\surveys\\all_ground_points_UTM11N_uid_flagged_cover.csv"
for rr in resolution:
    pts_file_in = initial_pts_file
    pts_file_out = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\dhs\\all_ground_points_dhs_r" + rr + ".csv"
    for ii in range(0, date.__len__()):
        ddi = date[ii]
        for jj in range(ii + 1, date.__len__()):
            ddj = date[jj]

            ras_sample = path_sub([dhs_dir_template, dhs_file_template], ddi=ddi, ddj=ddj, rr=rr)
            colname = str(ddi) + '-' + str(ddj)
            rastools.csv_sample_raster(ras_sample, pts_file_in, pts_file_out, "xcoordUTM11", "ycoordUTM11", colname, sample_no_data_value='')
            pts_file_in = pts_file_out
