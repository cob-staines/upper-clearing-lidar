import gdal
from affine import Affine
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

chm_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_all_test\\OUTPUT_FILES\\CHM\\19_149_all_test_628000_564657pit_free_chm_.25m.bil"

# open raster file
chm = gdal.Open(chm_in, gdal.GA_ReadOnly)

# get metadata
gt = chm.GetGeoTransform()
proj = chm.GetProjection()
cols = chm.RasterXSize
rows = chm.RasterYSize
band = chm.GetRasterBand(1)
no_data = band.GetNoDataValue()

# get affine transformation
T0 = Affine.from_gdal(*chm.GetGeoTransform())
# transform affine to describe pixel center
T1 = T0 * Affine.translation(0.5, 0.5)

# values as array
charm = np.array(chm.ReadAsArray())

# close file
chm = None

pixel_area = np.array(T1*(0, 0)) - np.array(T1*(1, 1))
pixel_area = pixel_area[1] ** 2

# using min canopy radius of 1
min_radius = 0
min_area = np.pi*min_radius ** 2
min_pixel = int(min_area/pixel_area)

# # identify local max
# mask = charm != no_data
#
#
# # buffer of 2
# def shifter(r, c):
#     return charm[2:rows-3, 2:cols-3] > charm[2+r:rows-3+r, 2+c:cols-3+c]
#
# plt.imshow(shifter(-2,-2), interpolation='nearest')
#
# peaks = shifter(-1, 0) & shifter(1, 0) & shifter(0, -1) & shifter(0, 1) & shifter(-1, -1) & shifter(-1, 1) & shifter(1, -1) & shifter(1, 1)
#
# # transform indices to (-1,-1)
# comp = charm[1:rows-2, 1:cols-2]
#
# comp_l = comp > charm[0:rows-3, 1:cols-2]
# comp_r = comp > charm[2:rows-1, 1:cols-2]
# comp_u = comp > charm[1:rows-2, 0:cols-3]
# comp_d = comp > charm[1:rows-2, 2:cols-1]
#
# peaks = comp_l & comp_r & comp_u & comp_d
#
# plt.imshow(peaks, interpolation='nearest')
# plt.show()


##

seed = charm > 2
peakmap = np.full([rows, cols, 2], np.nan)

# for each pixel
for ii in range(1, rows-2):
    for jj in range(1, cols-2):
        x = ii
        y = jj
        if seed[ii, jj]:
            moved = True
            while moved:
                window = charm[x-1:x+2, y-1:y+2]
                dx, dy = np.subtract(np.unravel_index(window.argmax(), window.shape), (1, 1))
                x = x + dx
                y = y + dy
                if (dx == 0) & (dy == 0):
                    moved = False
                    peakmap[ii, jj, :] = (x, y)
                if (x == 0) or (x == rows-1) or (y == 0) or (y == cols-1):
                    moved = False

peaklist = pd.DataFrame({'peak_x': peakmap[:, :, 0].flatten(),
                   'peak_y': peakmap[:, :, 1].flatten()})
peakcount = peaklist.groupby(['peak_x', 'peak_y']).size().reset_index().rename(columns={0: 'area'})

# threshold
peak_over_thresh = peakcount[peakcount.area >= min_pixel]

# merge index with map to visualize crowns captured
# assign index


peak_over_thresh['color'] = pd.Series(np.random.randint(255, size=len(peak_over_thresh.index)), index=peak_over_thresh.index)
peak_over_thresh['new_index'] = pd.Series(range(1, len(peak_over_thresh.index) + 1))
# highlight peaks

# merge with peaklist with
# isolate id
# refold and display
remap = pd.merge(peaklist, peak_over_thresh, how="left", on=("peak_x", "peak_y"))

colormap = np.array(remap.color).reshape((rows, cols))

plt.imshow(colormap, interpolation='nearest')

# output colormap
output_fname = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_all_test\\OUTPUT_FILES\\CHM\\canopy_color.tif"

outdriver = gdal.GetDriverByName("GTiff")
outdata = outdriver.Create(output_fname, cols, rows, 1, gdal.GDT_UInt16)
# Set metadata
outdata.SetGeoTransform(gt)
outdata.SetProjection(proj)

# Write data
outdata.GetRasterBand(1).WriteArray(colormap)
outdata.GetRasterBand(1).SetNoDataValue(0)
outdata = None

plt.hist(peakcount.area, 50)

peakcount['id'] = pd.Series(np.random.randint(255, size=len(peakcount.index)), index=peakcount.index)
