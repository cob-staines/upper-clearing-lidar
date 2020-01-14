import gdal
import rasterio
import ogr

rasin = """C:\\Users\\Cob\\index\\educational\\usask\\research\\dronefest\\data\\19_133_sd_analysis\\sfm_minu_lidar.tif"""
shpin = """C:\\Users\\Cob\\index\\educational\\usask\\research\\dronefest\\data\\19_133_sd_analysis\\DroneFest_surface_types_WGS84UTMzone13N.shp"""

# read input raster
ras = gdal.Open(rasin)
# get metadata
gt = ras.GetGeoTransform()
proj = ras.GetProjection()
cols = ras.RasterXSize
rows = ras.RasterYSize
band = ras.GetRasterBand(1)
no_data = band.GetNoDataValue()

shpdriver = ogr.GetDriverByName('ESRI Shapefile')
shpfile = shpdriver.Open(shpin, 0)
layer = shpfile.GetLayer(0)

output_fname = """C:\\Users\\Cob\\index\\educational\\usask\\research\\dronefest\\data\\19_133_sd_analysis\\DroneFest_surface_types_mask_WGS84UTMzone13N.tif"""

outdriver = gdal.GetDriverByName("GTiff")
outdata = outdriver.Create(output_fname, cols, rows, 1, gdal.GDT_UInt16)
# Set metadata
outdata.SetGeoTransform(gt)
outdata.SetProjection(proj)
# write data
gdal.RasterizeLayer(outdata, [1], layer, burn_values=[1])

# Set a no data value if required
outdata.GetRasterBand(1).SetNoDataValue(0)

outdata = None
