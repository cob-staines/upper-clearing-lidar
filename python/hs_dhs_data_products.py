import rastools

snow_on = ("19_045", "19_050", "19_052", "19_107", "19_123")
snow_off = "19_149"

resolutions = (".04", ".10", ".25", ".50", "1.00")

# snow depth products (HS)

# for each snow_on date
for date in snow_on:
    # for each resolution
    for res in resolutions:
        # snow_on
        ras_1_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\" + date + "\\" + date + "_snow_on\\OUTPUT_FILES\\DEM\\" + date + "_all_200311_628000_5646525dem_" + res + "m.bil"
        #snow_off
        ras_2_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\" + snow_off + "\\" + snow_off + "_snow_off\\OUTPUT_FILES\\DEM\\" + snow_off + "_all_200311_628000_5646525dem_" + res + "m.bil"

        # take difference (snow_off - snow_on)
        hs = rastools.ras_dif(ras_1_in, ras_2_in, inherit_from=2)
        # save output
        outfile = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\hs\\" + date + "\\hs_" + date + "_res_" + res + "m.tif"
        rastools.raster_save(hs, outfile, "GTiff")
        # burn snow mask
        snow_mask = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\site_library\\snow_depth_mask.shp"

        rastools.raster_burn(outfile, snow_mask, int(hs.no_data))
        # point sample
        pts_file = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\hs\\" + date + "\\all_ground_points_hs_" + date + ".csv"
        rastools.point_sample_raster(outfile, pts_file, pts_file, "xcoordUTM11", "ycoordUTM11", res, hs.no_data)

# differential snow depth products (dHS)
# for each snow_on date

date = snow_on
for ii in range(0, snow_on.__len__() - 1):
    # for each resolution
    for res in resolutions:
        # snow_earlier
        ras_1_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\" + date[ii] + "\\" + date[ii] + "_snow_on\\OUTPUT_FILES\\DEM\\" + date[ii] + "_all_200311_628000_5646525dem_" + res + "m.bil"
        #snow_later
        ras_2_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\" + date[ii + 1] + "\\" + date[ii + 1] + "_snow_on\\OUTPUT_FILES\\DEM\\" + date[ii + 1] + "_all_200311_628000_5646525dem_" + res + "m.bil"

        # take difference (snow_off - snow_on)
        hs = rastools.ras_dif(ras_1_in, ras_2_in, inherit_from=1)
        # save output
        outfile = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\dhs\\" + date[ii] + "-" + date[ii + 1] + "\\dhs_" + date[ii] + "-" + date[ii + 1] + "_res_" + res + "m.tif"
        rastools.raster_save(hs, outfile, "GTiff")
        # no burn snow mask
        # no point samples
