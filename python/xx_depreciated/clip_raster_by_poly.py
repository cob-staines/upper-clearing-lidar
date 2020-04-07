# get subprocess to call GDAL util

# define paths to raster and vector
inraster = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\DEM\\19_149_all_200311_628000_5646525dem_.10m.bil"
inshape = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\site_library\\snow_depth_mask.shp'


def raster_burn(ras_in, shp_in, burn_val):
    # burns "burn_val" into "ras_in" where overlaps with "shp_in"
    # !!burn_val must be string!!

    # Dependencies
    import subprocess

    # make gdal_rasterize command - will burn value 0 to raster where polygon intersects
    cmd = 'gdal_rasterize -burn ' + burn_val + ' ' + shp_in + ' ' + ras_in

    # run command
    subprocess.call(cmd, shell=True)
