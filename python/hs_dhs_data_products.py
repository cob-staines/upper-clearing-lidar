import rastools
import laslib
import time

date = ["19_045", "19_050", "19_052", "19_107", "19_123", "19_149"]
resolution = [".10", ".25", ".50", "1.00"]

dem_quantile = .20

las_in_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\<DATE>\\<DATE>_las_proc\\OUTPUT_FILES\\LAS\\<DATE>_classified_merged.las'
dem_ref_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\raster_templates\\hs_19_045_res_<RES>m.tif'
dem_path_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\<DATE>\\<DATE>_las_proc\\OUTPUT_FILES\\DEM\\<DATE>_dem_r<RES>m_q<QUANTILE>.tif'
count_path_template = dem_path_template.replace('q<QUANTILE>', 'count')
dhs_path_template = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\<DDI>_<DDJ>\\<DDI>_<DDJ>_dhs_r<RES>.tif'

# # create dem dictionary
# dem_dir = {}
# for dd in date:
#     dirdump = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\" + dd + "\\" + \
#                    dd + "_las_proc\\OUTPUT_FILES\\DEM\\"
#     dem_dir.update({dd: dirdump})

# create dems at all resolutions using quantile method
dem_path_q = dem_path_template.replace('<QUANTILE>', str(dem_quantile))

for dd in date:
    # update file paths with date
    las_in = las_in_template.replace('<DATE>', dd)
    dem_path_d = dem_path_q.replace('<DATE>', dd)
    count_path_d = count_path_template.replace('<DATE>', dd)
    for rr in resolution:
        # update file paths with resolution
        ras_template = dem_ref_template.replace('<RES>', rr)
        dem_path = dem_path_d.replace('<RES>', rr)
        count_path = count_path_d.replace('<RES>', rr)

        # calculate dem
        start = time.time()
        laslib.las_quantile_dem(las_in, ras_template, dem_quantile, q_out=dem_path, n_out=count_path)
        end = time.time()
        print(end - start)

for ii in range(0, date.__len__()):
    ddi = date[ii]
    for jj in range(ii, date.__len__()):
        ddj = date[jj]



# create merged dem products
ras_in_ext = ".bil"
for rr in resolution:
    # snow_off
    dd = snow_off
    ras_in_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\" + dd + "\\" + \
                 dd + "_snow_off\\TEMP_FILES\\12_dem\\res_" + rr + "\\"
    ras_out_path = dem_dir[dd] + dd + "_dem_res_" + rr + "m.bil"
    rastools.raster_merge(ras_in_dir, ras_in_ext, ras_out_path)

    # snow_on
    for dd in snow_on:
        ras_in_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\" + dd + "\\" + \
                     dd + "_snow_on\\TEMP_FILES\\12_dem\\res_" + rr + "\\"
        ras_out_path = dem_dir[dd] + dd + "_dem_res_" + rr + "m.bil"
        rastools.raster_merge(ras_in_dir, ras_in_ext, ras_out_path)

# snow depth products (HS)
# for each snow_on date
for rr in resolution:
    # for each resolution
    for dd in snow_on:

        # snow_on
        ras_1_in = dem_dir[dd] + dd + "_dem_res_" + rr + "m.bil"
        #snow_off
        ras_2_in = dem_dir[snow_off] + snow_off + "_dem_res_" + rr + "m.bil"

        # take difference (snow_off - snow_on)
        hs = rastools.ras_dif(ras_1_in, ras_2_in, inherit_from=2)

        # save output
        outfile = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\hs\\" + \
                  dd + "\\hs_" + dd + "_res_" + rr + "m.tif"
        rastools.raster_save(hs, outfile, "GTiff", data_format="float32")

        # burn snow_off mask into file
        snow_mask = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\site_library\\snow_depth_mask.shp"
        rastools.raster_burn(outfile, snow_mask, int(hs.no_data))

# differential snow depth products (dHS)
# for each snow_on date

date = snow_on.copy()
date.append(snow_off)
for ii in range(0, date.__len__() - 1):
    # for each resolution
    for rr in resolution:
        # snow_later
        ras_1_in = dem_dir[date[ii + 1]] + date[ii + 1] + "_dem_res_" + rr + "m.bil"

        # snow_earlier
        ras_2_in = dem_dir[date[ii]] + date[ii] + "_dem_res_" + rr + "m.bil"

        # take difference (snow_off - snow_on)
        hs = rastools.ras_dif(ras_1_in, ras_2_in, inherit_from=1)
        # save output
        outfile = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\dhs\\" + \
                  date[ii] + "-" + date[ii + 1] + "\\dhs_" + date[ii] + "-" + date[ii + 1] + "_res_" + rr + "m.tif"
        rastools.raster_save(hs, outfile, "GTiff")
        # no burn snow mask


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
