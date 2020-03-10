import gdal
import rasterio
import ogr
import numpy as np
from scipy.ndimage import convolve
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# config
elev_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_all_test\\OUTPUT_FILES\\CHM\\19_149_all_test_628000_564657pit_free_chm_.25m.bil"
canopy_min_elev = 2
kernel_dim = 3  # step size = (kernel_dim - 1)/2
max_scan = 30  # max number of steps
output_fname = prominence_out = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_all_test\\OUTPUT_FILES\\CHM\\19_149_all_test_628000_564657pit_free_chm_.25m_DCE.tiff"


# open single band geo-raster file
ras = gdal.Open(elev_in, gdal.GA_ReadOnly)

# get metadata
gt = ras.GetGeoTransform()
proj = ras.GetProjection()
cols = ras.RasterXSize
rows = ras.RasterYSize
band = ras.GetRasterBand(1)
no_data = band.GetNoDataValue()

# values as array
elev = np.array(ras.ReadAsArray())

# close file
ras = None

# define canopy binary
canopy = np.full([rows, cols], 0)
canopy[elev > canopy_min_elev] = 1

# preallocate distance to canopy edge (DCE) record
record = np.full([rows, cols], no_data)

kernel = np.full([kernel_dim, kernel_dim], 1)

binary = canopy.copy()
#while scan:
for ii in range(1, max_scan):
    convolved = convolve(binary, kernel)
    contenders = np.max([binary * (kernel_dim ** 2), convolved], 0)
    edges = (contenders > 0) & (contenders < kernel_dim ** 2)
    binary[edges] = 1
    record[edges] = ii

binary = 1 - canopy
for jj in range(1, max_scan):
    ii = 1 - jj
    convolved = convolve(binary, kernel)
    contenders = np.max([binary * (kernel_dim ** 2), convolved], 0)
    edges = (contenders > 0) & (contenders < kernel_dim ** 2)
    binary[edges] = 1
    record[edges] = ii


# output distance_to_canopy_map
outdriver = gdal.GetDriverByName("GTiff")
outdata = outdriver.Create(output_fname, cols, rows, 1, gdal.GDT_Int16)
# Set metadata
outdata.SetGeoTransform(gt)
outdata.SetProjection(proj)

# Write data
outdata.GetRasterBand(1).WriteArray(record)
outdata.GetRasterBand(1).SetNoDataValue(no_data)
outdata = None
