import numpy as np
import pandas as pd
from libraries import raslib
from scipy.ndimage.measurements import label, maximum_position

import time
start = time.time()

# config
elev_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\CHM\\19_149_all_200311_628000_564652_chm_.25m.bil"

# output file naming conventions
output_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\DFT\\"
file_base = elev_in.split("\\")[-1].replace(".bil", "")
treetops_out = output_dir + file_base + "_prom_treetops.csv"
nearest_out = output_dir + file_base + "_prom_nearest.tif"
distance_out = output_dir + file_base + "_prom_distance_to_tree.tif"

# parameters
z_min = 2  # lower elevation limit in elevation units (all cells below will be masked out of search)
z_step = 0.25  # vertical resolution of prominence calculation in elevation units
prominence_cutoff = 0  # peaks with prominence less than cutoff will be dropped (cutoff in units of elevation)
peak_elev_min = z_min + max([z_step, prominence_cutoff])  # peaks below will be dropped, in units of elevation
prominence_max = 50  # in units of elevation, multiple of z_step is ideal. To ignore, set above topography height
isolated_parent_id = -1111  # parent_id used when no parent found down to z_min.
prominence_max_parent_id = -2222  # parent_id used when no parent found within prominence_max range.

##
# load raster
ras = raslib.raster_load(elev_in)
elev = ras.data.copy()  # rename for legible coding

# define mask of valid data above  z_min
mask = (elev != ras.no_data) & (elev >= z_min)
# add 1-pixel buffer to mask
mask[0, :] = False
mask[ras.rows-1, :] = False
mask[:, 0] = False
mask[:, ras.cols-1] = False


def shifter(elev_temp, ras, r, c):
    # find peaks within 9-neighborhood
    # only works if no_data value is less than all other possible values (ex. -9999)
    return elev_temp[1:ras.rows-2, 1:ras.cols-2] >= elev_temp[1+r:ras.rows-2+r, 1+c:ras.cols-2+c]


peaks = np.full([ras.rows, ras.cols], False)
peaks[1:ras.rows-2, 1:ras.cols-2] = shifter(elev, ras, -1, 0) & shifter(elev, ras, 1, 0) & shifter(elev, ras, 0, -1) &\
                            shifter(elev, ras, 0, 1) & shifter(elev, ras, -1, -1) & shifter(elev, ras, -1, 1) &\
                            shifter(elev, ras, 1, -1) & shifter(elev, ras, 1, 1)

peakmask = (elev >= peak_elev_min)

peaks = peaks & mask & peakmask

# assign peaks id
peaks_xy = np.where(peaks)

peaklist = pd.DataFrame({'peak_x': peaks_xy[0],
                        'peak_y': peaks_xy[1]})

# drop peaks below z_min
peaklist.loc[:, "elev"] = elev[peaks_xy]
peaklist.loc[:, "id"] = peaklist.index

# build topo_elev list counting up from z_min (inclusive) to z_max (exclusive) by z_step, increasing order
z_max = np.max(elev[mask])
elev_band_count = int((z_max-z_min)/z_step)
topo_elev_max = z_min + elev_band_count*z_step
topo_elev = np.linspace(z_min - prominence_cutoff, topo_elev_max, elev_band_count + 1)

# peak band
# for each topo_elev from bottom
# for all peaks above layer
# set topo band up one

# build connected topos
band_below_tray = np.full_like(elev, -1).astype(int)
structure = np.ones((3, 3), dtype=np.int)  # connectivity matrix (9 neighborhood)
topo_con = np.full([ras.rows, ras.cols, topo_elev.__len__()], 0)
ncomponents = np.full_like(topo_elev, 0).astype(int)
for ll in range(0, topo_elev.__len__()):
    topo = (elev > topo_elev[ll]) & mask
    band_below_tray[topo] = ll
    topo_con[:, :, ll], ncomponents[ll] = label(topo, structure)

peaklist.loc[:, "band_below"] = band_below_tray[peaks_xy]

# for each topo_con layer (starting from top, moving down
col_elev_tray = elev.copy()
for ll in range(len(topo_elev) - 1, -1, -1):
    # find maximum within each label
    comp_peaks = maximum_position(elev, labels=topo_con[:, :, ll], index=list(range(1, ncomponents[ll] + 1)))
    still_peaks = tuple(np.array(list(zip(*comp_peaks))))
    col_elev_tray[still_peaks] = topo_elev[ll]
# record layer elevation in peaklist as "min peak elev"
# when all layers are done, determine prominence by "min peak elev" - peak elev
peaklist.loc[:, "col_elev"] = col_elev_tray[peaks_xy]
peaklist.loc[:, "prom_expected"] = peaklist.elev - (peaklist.col_elev - z_step/2)


# can we determine who is the parent?
# what is peak of label below "min peak elev"
# do we need this? no.

from scipy import spatial
# peaktree = np.array([peaks_xy[0], peaks_xy[1]]).transpose()
# tree = spatial.KDTree(peaktree, leafsize=10)
# tree.query([0, 0])

peaklist.loc[:, "iso_expected"] = 0



# how do we determine isolation?
# ugly: threshold above peak, find nearest neighbor
# cleaner: go through all layers again
for ll in range(1, len(topo_elev)):
    above_thresh = np.array(list(zip(*np.where(elev > topo_elev[ll]))))
    tree = spatial.KDTree(above_thresh, leafsize=10)
    query_list = np.array(list(zip(*[peaklist.peak_x[peaklist.band_below == ll-1], peaklist.peak_y[peaklist.band_below == ll-1]])))
    distance, index = tree.query(query_list)
    peaklist.loc[peaklist.band_below == ll-1, "iso_expected"] = distance
    print(ll)
    # build nntree for each threshold, seed with peaks just below layer

# interpret prominence_max to layers (floor)
prominence_max_steps = int(prominence_max/z_step)

# find parents
parentlist = peaklist.copy()
parentlist.loc[:, "parent_id"] = -9999
parentlist.loc[:, "prominence_expected"] = -9999

# for each peak in peaklist
#def test_fun():
#for peak_id in range(10000, 10100):
for peak_id in range(0, len(peaklist)):

    # get peak coordinates
    peak_coords = (peaklist.peak_x[peak_id], peaklist.peak_y[peak_id])
    peak_elev = elev[peak_coords]

    # begin with topo_elev band just above prominence_cutoff
    topo_band_init = np.where(topo_elev < elev[peak_coords] - prominence_cutoff)[0][-1] + 1

    # increment through topo elev bands beginning from topo_band_init with incrementer ll
    # until parent is found, or prominence_max_steps is reached
    ll = 0
    found_parent = False
    while not found_parent:
        # set topo elevation band
        elev_band = topo_band_init - ll

        # should search be initialized?
        if ll > prominence_max_steps:
            # if prominence_max_steps reached

            found_parent = True
            # record parent as beyond prominence max
            parentlist.loc[peak_id, "parent_id"] = prominence_max_parent_id
            # write prominence_max (be careful not to misinterpret here!)
            parentlist.loc[peak_id, "prominence_expected"] = prominence_max
        elif elev_band < 0:
            # if elev_band out of range

            found_parent = True
            # record parent as isolated (up to z_min)
            parentlist.loc[peak_id, "parent_id"] = isolated_parent_id
            # write expected prominence: mean of upper and lower limits
            parentlist.loc[peak_id, "prominence_expected"] = elev[peak_coords] - (min([elev[peak_coords], topo_elev[elev_band + 2]]) + topo_elev[elev_band + 1]) / 2
        else:
            # check connected neighborhood of next elevation band for higher peak
            neighborhood_id = topo_con[peak_coords[0], peak_coords[1], elev_band]
            neighbor_list = np.where((topo_con[:, :, elev_band] == neighborhood_id) & peaks)
            neighbor_elev = elev[neighbor_list]

            if peak_elev < np.max(neighbor_elev):
                # if higher peak exists in neighborhood
                found_parent = True

                # find parent_id
                parent_arg = np.argmax(neighbor_elev)
                parent_coords = (neighbor_list[0][parent_arg], neighbor_list[1][parent_arg])
                parent_id = int(peaklist.id[(peaklist.peak_x == parent_coords[0]) & (peaklist.peak_y == parent_coords[1])])

                # write parent_id to parentlist
                parentlist.loc[peak_id, "parent_id"] = parent_id
                # write expected prominence: mean of upper and lower limits
                parentlist.loc[peak_id, "prominence_expected"] = elev[peak_coords] - (
                        min([elev[peak_coords], topo_elev[elev_band + 1]]) + topo_elev[elev_band]) / 2

        # increment to next elevation band
        ll = ll+1
end = time.time()
print(end - start)

# calculate geo-coords
UTM_coords = ras.T1 * [parentlist.peak_y, parentlist.peak_x]
parentlist.loc[:, "UTM11N_x"] = UTM_coords[0]
parentlist.loc[:, "UTM11N_y"] = UTM_coords[1]

# write parentlist to file
output = parentlist.copy()
output = output.drop(["peak_x", "peak_y"], axis=1)
output.to_csv(treetops_out, index=False)

# calculate distance from tree
parent_filtered = parentlist.copy()
parent_filtered = parent_filtered.loc[parent_filtered.prominence_expected >= prominence_cutoff]
parent_filtered = parent_filtered.reset_index(drop=True)
# preallocate
nearest_map = np.full_like(elev, ras.no_data)
distance_map = np.full_like(elev, np.nan)
# slow but works (60s?)
for ii in range(0, ras.cols):
    for jj in range(0, ras.rows):
        cell_coords = ras.T1 * [ii, jj]
        distances = np.sqrt((cell_coords[0] - np.array(parent_filtered.UTM11N_x))**2 + (cell_coords[1] - np.array(parent_filtered.UTM11N_y))**2)
        nearest_id = np.argmin(distances)
        nearest_map[jj, ii] = parent_filtered.id[nearest_id]
        distance_map[jj, ii] = distances[nearest_id]


# export parent_map to raster file
ras_nearest = ras
ras_nearest.data = nearest_map
raslib.raster_save(ras_nearest, nearest_out, data_format="int")

# export distance_map to raster file
ras_distance = ras
ras_distance.data = distance_map
raslib.raster_save(ras_distance, distance_out, data_format="float")

end = time.time()
print(end - start)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.imshow(band_below_tray, interpolation='nearest')