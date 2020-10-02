from osgeo import gdal, gdalconst
import numpy as np

# Source
src_filename = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\qc_test\\19_149_dem_r.25m_count_crop1.tif'
src = gdal.Open(src_filename, gdalconst.GA_ReadOnly)
src_proj = src.GetProjection()
src_geotrans = src.GetGeoTransform()

# We want a section of source that matches this:
match_filename = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\qc_test\\19_149_dem_r.25m_count_crop2.tif'
match_ds = gdal.Open(match_filename, gdalconst.GA_ReadOnly)
match_proj = match_ds.GetProjection()
match_geotrans = match_ds.GetGeoTransform()
wide = match_ds.RasterXSize
high = match_ds.RasterYSize

# # Output / destination
# dst_filename = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\qc_test\\19_149_dem_r.25m_count_crop2_reprj.tif'
# dst = gdal.GetDriverByName('GTiff').Create(dst_filename, wide, high, 1, gdalconst.GDT_Float32)
# dst.SetGeoTransform(match_geotrans)
# dst.SetProjection(match_proj)

mem_drv = gdal.GetDriverByName('MEM')
dest = mem_drv.Create('', wide, high, 1, gdal.GDT_Float32)
dest.GetRasterBand(1).WriteArray(np.full((high, wide), np.nan), 0, 0)
# Set the geotransform
dest.SetGeoTransform(match_geotrans)
dest.SetProjection(match_proj)
# Perform the projection/resampling
res = gdal.ReprojectImage(src, dest, src_proj, match_proj, gdal.GRA_Bilinear)

ofnp = myarray = np.array(dest.GetRasterBand(1).ReadAsArray())

# # Do the work
# gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_Bilinear)
# del dst # Flush



