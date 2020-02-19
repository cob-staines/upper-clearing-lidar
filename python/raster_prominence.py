import gdal
import rasterio
import ogr
from affine import Affine
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

# single band geo-raster file
elev_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_all_test\\OUTPUT_FILES\\CHM\\19_149_all_test_628000_564657pit_free_chm_.25m.bil"
prominence_out = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_all_test\\OUTPUT_FILES\\CHM\\19_149_all_test_628000_564657pit_free_chm_.25m_prominence.csv"

# open raster file
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

# define mask of valid data
mask = elev != no_data

# add 1 pixel buffer
mask[0, :] = False
mask[rows-1, :] = False
mask[:, 0] = False
mask[:, cols-1] = False

# if no_mask is between min and max, error!

# find peaks within 9-neighborhood
# buffer of 1


def shifter(ras, r, c):
    return ras[1:rows-2, 1:cols-2] > ras[1+r:rows-2+r, 1+c:cols-2+c]


peaks = np.full([rows, cols], False)
peaks[1:rows-2, 1:cols-2] = shifter(elev, -1, 0) & shifter(elev, 1, 0) & shifter(elev, 0, -1) & shifter(elev, 0, 1) & shifter(elev, -1, -1) & shifter(elev, -1, 1) & shifter(elev, 1, -1) & shifter(elev, 1, 1)
peaks = peaks & mask

# assign peaks id
peaks_xy = np.where(peaks)
peaklist = pd.DataFrame({'peak_x': peaks_xy[0],
                   'peak_y': peaks_xy[1]})

peak_elev_cutoff = 10  # in units of elevation

# filter peaks by elevation cutoff
peaklist.loc[:, "elev"] = elev[peaks_xy]
peaklist = peaklist.loc[peaklist.elev > peak_elev_cutoff]
peaklist = peaklist.reset_index(drop=True)
peaklist.loc[:, "id"] = peaklist.index

z_step = 0.5  # in elevation units
z_min = np.min(elev[mask])
z_max = np.max(elev[mask])

# elevation list built with elevation step counting down from z_max (exclusive) to z_min (exclusive)
layer_count = int((z_max-z_min)/z_step)
elevation_min = z_max - layer_count*z_step
elevation_max = z_max - z_step
topo_elev = np.linspace(elevation_min, elevation_max, layer_count)

# build topo with heights at topo_elevs
topo = np.full([rows, cols, topo_elev.__len__()], False)
for ii in range(0, topo_elev.__len__()-1):
    topo[:, :, ii] = (elev > topo_elev[ii]) & mask

# for each peak

# begin with layer with elevation just below peak

# find neighborhood

# are any other peaks in neighborhood?
    #if so, are they greater than this peak?
        #if so, record id as parent and end loop
    #if not move down one layer and repeat

cutoff = 3  # in units of elevation, multiple of step ideal
layer_cutoff = int(cutoff/z_step)

# find parents
parent_list = peaklist.copy()
parent_list.loc[:, "parent_id"] = -9999
parent_list.loc[:, "prominence_lower_lim"] = -9999

# for each peak
for peak_id in range(0, peaklist.__len__()):
    peak_coords = (peaklist.peak_x[peak_id], peaklist.peak_y[peak_id])

    found_parent = False
    topo_band_below = np.where(topo_elev < elev[peak_coords])[0][-1]

    ll = 0
    # for elevation band through cutoff, or until parent is found
    while ~found_parent & (ll < layer_cutoff):
        elev_band = topo_band_below - ll

        # list of cells to search
        if elev_band > 0:
            # initialize search list with peak only
            searchlist = [peak_coords]
        else:
            # do not search if elevation band is out of bounds
            searchlist = []
        # report of connected cells
        connectedlist = []
        # map of searched cells
        queued = np.full([rows, cols], False)
        queued[peak_coords] = True

        # while items remain in searchlist
        while searchlist.__len__() > 0:
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

        if connectedlist.__len__() > 0:
            # take unique elements in connectedlist
            connectedlist = list(zip(*connectedlist))
            # convert to data frame
            connected_df = pd.DataFrame({'neighbor_x': connectedlist[0],
                            'neighbor_y': connectedlist[1]})
            # merge with peaklist
            peak_neighbors = pd.merge(peaklist, connected_df, how="inner", left_on=("peak_x", "peak_y"), right_on=("neighbor_x", "neighbor_y"))
            # define parents as neighboring peaks with elevation greater than peak
            parent = peak_neighbors.loc[peak_neighbors.elev > elev[peak_coords]]
            parent = parent.reset_index()
            if parent.__len__() > 0:
                if parent.__len__() > 1:
                    # use highest parent
                    parent = parent.iloc[np.argmax(np.array(parent.elev))]
                # write parent_id to parent_list
                parent_list.parent_id.loc[peak_id] = int(parent.id)
                found_parent = True

                # write lower limit of prominence
                parent_list.prominence_lower_lim.loc[peak_id] = elev[peak_coords] - topo_elev[elev_band + 1]

                # could also write area below?

        ll = ll+1

output = parent_list.copy()
output_coords = T1 * [parent_list.peak_x, parent_list.peak_y]
output.loc[:, "UTM11N_x"] = output_coords[0]
output.loc[:, "UTM11N_y"] = output_coords[1]
output = output.drop(["peak_x", "peak_y"], axis=1)
output.to_csv(prominence_out, index=False)

plt.imshow(queued, interpolation='nearest')
