import rastools
import laslib
import time
import os

date = ["19_045", "19_050", "19_052", "19_107", "19_123", "19_149"]
int_date = ["19_149"]
resolution = [".04", ".10", ".25", ".50", "1.00"]

dem_quantile = .25
interpolation_threshold = 0

las_in_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\<DATE>\\<DATE>_las_proc\\OUTPUT_FILES\\LAS\\<DATE>_las_proc_classified_merged.las'
dem_ref_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\raster_templates\\hs_19_045_res_<RES>m.tif'
dem_dir_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\<DATE>\\<DATE>_las_proc\\OUTPUT_FILES\\DEM\\'
dem_file_template = '<DATE>_dem_r<RES>m_q<QUANT>.tif'
count_file_template = '<DATE>_dem_r<RES>m_count.tif'

dhs_dir_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\dhs\\<DDI>-<DDJ>\\'
dhs_file_template = '<DDI>-<DDJ>_dhs_r<RES>.tif'

int_dir_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\<DATE>\\<DATE>_las_proc\\OUTPUT_FILES\\DEM\\interpolated\\'
int_file_template = '<DATE>_dem_r<RES>m_q<QUANT>_interpolated_t<ITN>.tif'

def path_sub(path, dd=None, rr=None, qq=None, ddi=None, ddj=None, itn=None):
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
        if min is not None:
            path[ii] = path[ii].replace('<ITN>', str(itn))

    return ''.join(path)


# create dems at all resolutions using quantile method
for dd in date:
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

# differential dem products
for ii in range(0, date.__len__()):
    ddi = date[ii]
    for jj in range(ii + 1, date.__len__()):
        ddj = date[jj]

        # create dHS directory if does not exist
        dhs_dir = path_sub(dhs_dir_template, ddi=ddi, ddj=ddj)
        if not os.path.exists(dhs_dir):
            os.makedirs(dhs_dir)

        for rr in resolution:
            ddi_in = path_sub([dem_dir_template, dem_file_template], dd=ddi, rr=rr, qq=dem_quantile)
            ddj_in = path_sub([dem_dir_template, dem_file_template], dd=ddj, rr=rr, qq=dem_quantile)
            dhs_out = path_sub([dhs_dir_template, dhs_file_template], ddi=ddi, ddj=ddj, rr=rr)

            dhs = rastools.raster_dif(ddi_in, ddj_in, inherit_from=2, dif_out=dhs_out)

# interpolated products
for dd in int_date:
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

#### BOOKMARK -- OLD HAT BELOW

# point sample HS products
initial_pts_file = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\surveys\\all_ground_points_UTM11N_uid_flagged_cover.csv"
for dd in snow_on:
    pts_file_in = initial_pts_file
    pts_file_out = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\hs\\" + \
                   dd + "\\all_ground_points_hs_" + dd + ".csv"
    for rr in resolution:
        ras_sample = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\hs\\" + \
                     dd + "\\hs_" + dd + "_res_" + rr + "m.tif"
        rastools.point_sample_raster(ras_sample, pts_file_in, pts_file_out, "xcoordUTM11", "ycoordUTM11", rr)
        pts_file_in = pts_file_out
