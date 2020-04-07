# import rasterio
# import ogr


def raster_load(ras_in):
    # takes in path for georaster file, returns raster objects with following attributes:
        # ras.data -- numpy array of raster data
        # ras.gt -- geotransformation
        # ras.proj -- projection
        # ras.cols -- number of columns in raster
        # ras.rows -- number of rows in raster
        # ras.band -- raster band (1 only supported)
        # ras.T0 -- affine transformation for cell corners
        # ras.T1 -- affine transformation for cell centers

    # dependencies
    import gdal
    from affine import Affine
    import numpy as np

    #define class rasterObj
    class rasterObj(object):
        def __init__(self, raster):
            # load metadata
            self.gt = raster.GetGeoTransform()
            self.proj = raster.GetProjection()
            self.cols = raster.RasterXSize
            self.rows = raster.RasterYSize
            self.band = raster.GetRasterBand(1)
            self.no_data = self.band.GetNoDataValue()
            # get affine transformation
            self.T0 = Affine.from_gdal(*raster.GetGeoTransform())
            # cell-centered affine transformation
            self.T1 = self.T0 * Affine.translation(0.5, 0.5)
            # load data
            self.data = np.array(raster.ReadAsArray())

    # open single band geo-raster file
    ras = gdal.Open(ras_in, gdal.GA_ReadOnly)

    # read data
    ras_out = rasterObj(ras)

    # close file
    ras = None

    return ras_out


# saves raster to file
def raster_save(ras_object, file_path, file_format):
    # saves "ras_object" to "file_path" in "file_format"
    # format can be: "GTiff",

    # dependencies
    import gdal

    outdriver = gdal.GetDriverByName(file_format)
    outdata = outdriver.Create(file_path, ras_object.cols, ras_object.rows, 1, gdal.GDT_Float32)
    # Set metadata
    outdata.SetGeoTransform(ras_object.gt)
    outdata.SetProjection(ras_object.proj)

    # Write data
    outdata.GetRasterBand(1).WriteArray(ras_object.data)
    outdata.GetRasterBand(1).SetNoDataValue(ras_object.no_data)


def ras_dif(ras_1_in, ras_2_in, inherit_from=1):
    # Returns raster object as follows:
        # ras_dif.data = ras_1. data - ras_2.data
        # metadata inherited from "inherit_from" (1 or 2).
    # Rasters must be identical in location, scale, and size

    # Dependencies
    import numpy as np

    if inherit_from == 1:
        ras_A = ras = raster_load(ras_1_in)
        ras_B = ras = raster_load(ras_2_in)
        flip_factor = 1
    elif inherit_from == 2:
        ras_A = ras = raster_load(ras_2_in)
        ras_B = ras = raster_load(ras_1_in)
        flip_factor = -1
    else:
        raise Exception('"inherit_from" must be either "1" or "2."')

    # check if identical origins and scales
    aff_dif = np.array(ras_A.T1) - np.array(ras_B.T1)

    if np.sum(np.abs(aff_dif)) != 0:
        raise Exception('Rasters are of different scales or origins, no difference was taken. Cell shifting may be needed!')

    ras_dif = ras_A
    # handle nas!
    mask = (ras_A.data == ras_A.no_data) | (ras_B.data == ras_B.no_data)
    ras_dif.data = (ras_A.data - ras_B.data) * flip_factor
    ras_dif.data[mask] = ras_A.no_data

    return ras_dif


def raster_burn(ras_in, shp_in, burn_val):
    # burns "burn_val" into "ras_in" where overlaps with "shp_in"
    # !!burn_val must be string!!

    # Dependencies
    import subprocess

    # convert any numbers to string
    burn_val = str(burn_val)

    # make gdal_rasterize command - will burn value 0 to raster where polygon intersects
    cmd = 'gdal_rasterize -burn ' + burn_val + ' ' + shp_in + ' ' + ras_in

    # run command
    subprocess.call(cmd, shell=True)


def point_sample_raster(ras_in, pts_in, pts_out, pts_xcoord_name, pts_ycoord_name, sample_col_name, sample_no_data_value):

    # takes in csv file of points "pts_in" with x-column "pts_xcoord_name" and y-column "pts_ycoord_name" and saves
    # point values of raster "ras_in" to column "sample_col_name" in output csv "pts_out"

    # Dependencies
    import pandas as pd
    import numpy as np

    # read points
    pts = pd.read_csv(pts_in)
    # read raster
    ras = raster_load(ras_in)

    # convert point coords to raster index
    row_col_pts = np.rint(~ras.T1 * (pts[pts_xcoord_name], pts[pts_ycoord_name])).astype(int)
    row_col_pts = (row_col_pts[1], row_col_pts[0])

    # read raster values of points
    samples = ras.data[row_col_pts]

    # replace no_data values
    samples[samples == ras.no_data] = np.nan

    # add to pts df
    pts.loc[:, sample_col_name] = samples

    # write to file
    pts.to_csv(pts_out, index=False, na_rep=sample_no_data_value)
