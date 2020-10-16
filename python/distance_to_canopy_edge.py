import numpy as np
from scipy.ndimage import convolve
import rastools
import os

# config
ras_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\CHM\\19_149_spike_free_chm_r.10m.bil"
step_size = 0.10  # in m
canopy_min_elev = 2
kernel_dim = 3  # step size = (kernel_dim - 1)/2
max_scan = 30  # max number of steps
file_out = ras_in.replace('CHM', 'DCE').replace('.bil', '_dce.tif')

# load raster
ras = rastools.raster_load(ras_in)

# define canopy binary
canopy = np.full([ras.rows, ras.cols], 0)
canopy[ras.data > canopy_min_elev] = 1

# preallocate distance to canopy edge (DCE) record
record = np.full([ras.rows, ras.cols], np.nan)

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

# correct for step size
record = record * step_size

record[np.isnan(record)] = ras.no_data

# export
if not os.path.exists(dir_out):
    os.makedirs(dir_out)

ras_dce = ras
ras_dce.data = record
rastools.raster_save(ras_dce, file_out, data_format="float32")
