def raster_load(ras_in):
    # dependencies
    import gdal
    import rasterio
    import ogr
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
