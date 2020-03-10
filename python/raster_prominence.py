import gdal
import rasterio
import ogr
from affine import Affine
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd


# config
elev_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_all_test\\OUTPUT_FILES\\CHM\\19_149_all_test_628000_564657pit_free_chm_.25m.bil"
prominence_out = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_all_test\\OUTPUT_FILES\\CHM\\19_149_all_test_628000_564657pit_free_chm_.25m_prominence.csv"

z_min = 2  # lower elevation limit in elevation units, all cells below will be masked out of search
z_step = 0.25  # resolution of prominence calculation in elevation units
peak_elev_min = z_min + z_step  # peaks below will be dropped, in units of elevation
prominence_max = 10  # in units of elevation, multiple of z_step is ideal. To ignore, set above topography height
isolated_parent_id = -1111  # parent_id used when no parent found down to z_min.
prominence_max_parent_id = -2222  # parent_id used when no parent found within prominence_max range.

# open single band geo-raster file
ras = gdal.Open(elev_in, gdal.GA_ReadOnly)

# get metadata
gt = ras.GetGeoTransform()
proj = ras.GetProjection()
cols = ras.RasterXSize
rows = ras.RasterYSize
band = ras.GetRasterBand(1)
no_data = band.GetNoDataValue()

# get affine transformation
T0 = Affine.from_gdal(*ras.GetGeoTransform())
# transform affine to describe pixel center
T1 = T0 * Affine.translation(0.5, 0.5)

# values as array
elev = np.array(ras.ReadAsArray())

# close file
ras = None

# define mask of valid data above  z_min
mask = (elev != no_data) & (elev >= z_min)
# add 1-pixel buffer to mask
mask[0, :] = False
mask[rows-1, :] = False
mask[:, 0] = False
mask[:, cols-1] = False

# find peaks within 9-neighborhood using buffer of 1



def shifter(ras, r, c):
    #set no_data values higher than max to throw out peaks with no_data neighbors
    high_no_data = int(np.max(ras) + 1)
    ras[ras == no_data] = high_no_data
    return ras[1:rows-2, 1:cols-2] > ras[1+r:rows-2+r, 1+c:cols-2+c]


peaks = np.full([rows, cols], False)
peaks[1:rows-2, 1:cols-2] = shifter(elev, -1, 0) & shifter(elev, 1, 0) & shifter(elev, 0, -1) &\
                            shifter(elev, 0, 1) & shifter(elev, -1, -1) & shifter(elev, -1, 1) &\
                            shifter(elev, 1, -1) & shifter(elev, 1, 1)
peaks = peaks & mask

# assign peaks id
peaks_xy = np.where(peaks)
peaklist = pd.DataFrame({'peak_x': peaks_xy[0],
                        'peak_y': peaks_xy[1]})

# drop peaks below z_min
peaklist.loc[:, "elev"] = elev[peaks_xy]
peaklist = peaklist.loc[peaklist.elev > peak_elev_min]
peaklist = peaklist.reset_index(drop=True)
peaklist.loc[:, "id"] = peaklist.index

# build topo_elev list counting up from z_min (inclusive) to z_max (exclusive) by z_step, increasing order
z_max = np.max(elev[mask])
elev_band_count = int((z_max-z_min)/z_step)
topo_elev_max = z_min + elev_band_count*z_step
topo_elev = np.linspace(z_min, topo_elev_max, elev_band_count + 1)

# build boolean topos: True if greater than topo_elevs & mask
topo = np.full([rows, cols, topo_elev.__len__()], False)
for ii in range(0, topo_elev.__len__()-1):
    topo[:, :, ii] = (elev > topo_elev[ii]) & mask

# interpret prominence_max to layers
prominence_max_steps = int(prominence_max/z_step)

# find parents
parentlist = peaklist.copy()
parentlist.loc[:, "parent_id"] = -9999
parentlist.loc[:, "prominence_expected"] = -9999

# for each peak
for peak_id in range(0, len(peaklist)):
    # get peak coordinates
    peak_coords = (peaklist.peak_x[peak_id], peaklist.peak_y[peak_id])

    # begin with topo_elev band immediately below peak
    topo_band_below = np.where(topo_elev < elev[peak_coords])[0][-1]

    # increment through topo elev bands beginning from topo_band below with incrementer ll
    # until parent is found, or prominence_max_steps is reached
    ll = 0
    found_parent = False
    while not found_parent:
        # set topo elevation band
        elev_band = topo_band_below - ll

        # should search be initialized?
        if ll > prominence_max_steps:
            # empty search to discontinue if prominence_max_steps reached
            searchlist = []

            found_parent = True
            # record parent as beyond prominence max
            parentlist.loc[peak_id, "parent_id"] = prominence_max_parent_id
            # write prominence_max (be careful here!)
            parentlist.loc[peak_id, "prominence_expected"] = prominence_max
        elif elev_band < 0:
            # empty search to discontinue if elev_band out of range
            searchlist = []

            found_parent = True
            # record parent as isolated (up to z_min)
            parentlist.loc[peak_id, "parent_id"] = isolated_parent_id
            # write expected prominence: mean of upper and lower limits
            parentlist.loc[peak_id, "prominence_expected"] = elev[peak_coords] - (min([elev[peak_coords], topo_elev[elev_band + 2]]) + topo_elev[elev_band + 1]) / 2
        else:
            # initialize parent search with peak if elev_band in range
            searchlist = [peak_coords]

        # report of connected cells
        connectedlist = []
        # map of searched cells
        queued = np.full([rows, cols], False)
        queued[peak_coords] = True

        # while items remain in searchlist
        while len(searchlist) > 0:
            searching = searchlist.pop(0)

            connected_patch = topo[(searching[0]-1):(searching[0]+2), (searching[1]-1):(searching[1]+2), elev_band]
            queued_patch = queued[(searching[0]-1):(searching[0]+2), (searching[1]-1):(searching[1]+2)]
            mask_patch = mask[(searching[0]-1):(searching[0]+2), (searching[1]-1):(searching[1]+2)]

            # neighbors connected, not yet searched, and in mask
            add_patch = connected_patch & ~queued_patch & mask_patch
            add_patch_list = np.where(add_patch)
            add_list = (add_patch_list[0] - 1 + searching[0], add_patch_list[1] - 1 + searching[1])
            add_tuples = list(zip(add_list[0], add_list[1]))

            searchlist.extend(add_tuples)
            connectedlist.extend(add_tuples)
            queued[add_list] = True

        if len(connectedlist) > 0:
            # take unique elements in connectedlist
            connectedlist = list(zip(*connectedlist))
            # convert to data frame
            connected_df = pd.DataFrame({'neighbor_x': connectedlist[0],
                            'neighbor_y': connectedlist[1]})
            # merge with peaklist
            peak_neighbors = pd.merge(peaklist, connected_df, how="inner",
                                      left_on=("peak_x", "peak_y"), right_on=("neighbor_x", "neighbor_y"))
            # define parents as neighboring peaks with elevation greater than peak
            parent = peak_neighbors.loc[peak_neighbors.elev > elev[peak_coords]]
            if len(parent) > 0:
                if len(parent) > 1:
                    # use highest parent
                    parent = parent.iloc[np.argmax(np.array(parent.elev))]

                found_parent = True
                # write parent_id to parentlist
                parentlist.loc[peak_id, "parent_id"] = int(parent.id)
                # write expected prominence: mean of upper and lower limits
                parentlist.loc[peak_id, "prominence_expected"] = elev[peak_coords] - (
                        min([elev[peak_coords], topo_elev[elev_band + 1]]) + topo_elev[elev_band])/2

        # increment to next elevation band
        ll = ll+1

# calculate coordinates
UTM_coords = T1 * [parentlist.peak_y, parentlist.peak_x]
parentlist.loc[:, "UTM11N_x"] = UTM_coords[0]
parentlist.loc[:, "UTM11N_y"] = UTM_coords[1]

# write parentlist to file
output = parentlist.copy()
output = output.drop(["peak_x", "peak_y"], axis=1)
output.to_csv(prominence_out, index=False)




# calculate distance from tree
prominence_cutoff = 1  # peaks with prominence less than cutoff will be dropped (cutoff in units of elevation)
parent_filtered = parentlist.copy()
parent_filtered = parent_filtered.loc[parent_filtered.prominence_expected >= prominence_cutoff]
parent_filtered = parent_filtered.reset_index(drop=True)

# preallocate
parent_map = np.full([rows, cols], no_data)
distance_map = np.full([rows, cols], np.nan)

# slow but works (60s?)
for ii in range(0, cols):
    for jj in range(0, rows):
        cell_coords = T1 * [ii, jj]
        distances = np.sqrt((cell_coords[0] - np.array(parent_filtered.UTM11N_x))**2 + (cell_coords[1] - np.array(parent_filtered.UTM11N_y))**2)
        nearest_id = np.argmin(distances)
        parent_map[jj, ii] = parent_filtered.id[nearest_id]
        distance_map[jj, ii] = distances[nearest_id]

# output distance_map
output_fname = prominence_out = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_all_test\\OUTPUT_FILES\\CHM\\19_149_all_test_628000_564657pit_free_chm_.25m_parent_dist.tiff"

outdriver = gdal.GetDriverByName("GTiff")
outdata = outdriver.Create(output_fname, cols, rows, 1, gdal.GDT_Float32)
# Set metadata
outdata.SetGeoTransform(gt)
outdata.SetProjection(proj)

# Write data
outdata.GetRasterBand(1).WriteArray(distance_map)
outdata.GetRasterBand(1).SetNoDataValue(no_data)
outdata = None

# output parent_map
output_fname = prominence_out = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_all_test\\OUTPUT_FILES\\CHM\\19_149_all_test_628000_564657pit_free_chm_.25m_parent_id.tiff"

outdriver = gdal.GetDriverByName("GTiff")
outdata = outdriver.Create(output_fname, cols, rows, 1, gdal.GDT_Int16)
# Set metadata
outdata.SetGeoTransform(gt)
outdata.SetProjection(proj)

# Write data
outdata.GetRasterBand(1).WriteArray(parent_map)
outdata.GetRasterBand(1).SetNoDataValue(no_data)
outdata = None

plt.imshow(distance_map, interpolation='nearest')