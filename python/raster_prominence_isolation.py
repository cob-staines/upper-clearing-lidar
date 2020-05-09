import numpy as np
import pandas as pd
import rastools
from scipy import spatial
from scipy.ndimage.measurements import label, maximum_position

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

##
# load raster
print("Loading raster")
ras = rastools.raster_load(elev_in)
elev = ras.data.copy()  # rename for legible coding

print("Identifying peaks")
# define mask of valid data above  z_min
mask = (elev != ras.no_data) & (elev >= z_min)
# add 1-pixel buffer to mask
mask[0, :] = False
mask[ras.rows-1, :] = False
mask[:, 0] = False
mask[:, ras.cols-1] = False


def shifter(elev_temp, ras_temp, r, c):
    # find peaks within 9-neighborhood
    # only works if no_data value is less than all other possible values (ex. -9999)
    return elev_temp[1:ras_temp.rows-2, 1:ras_temp.cols-2] >= elev_temp[1+r:ras_temp.rows-2+r, 1+c:ras_temp.cols-2+c]


peaks = np.full([ras.rows, ras.cols], False)
peaks[1:ras.rows-2, 1:ras.cols-2] = shifter(elev, ras, -1, 0) & shifter(elev, ras, 1, 0) & shifter(elev, ras, 0, -1) &\
                            shifter(elev, ras, 0, 1) & shifter(elev, ras, -1, -1) & shifter(elev, ras, -1, 1) &\
                            shifter(elev, ras, 1, -1) & shifter(elev, ras, 1, 1)

peaks = peaks & mask

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
topo_elev = np.linspace(z_min, topo_elev_max, elev_band_count + 1)

print("Building topo bands")
# build connected topos
band_below_tray = np.full_like(elev, -1).astype(int)  # record topo band just below each point (used in isolation calc)
structure = np.ones((3, 3), dtype=np.int)  # connectivity matrix for cols (9 neighborhood)
topo_con = np.full([ras.rows, ras.cols, topo_elev.__len__()], 0)  # array of labeled neighborhoods above each band
ncomponents = np.full_like(topo_elev, 0).astype(int)  # number of separate neighborhoods above each band
for ll in range(0, topo_elev.__len__()):
    # consider all points above elevation band
    topo = (elev > topo_elev[ll]) & mask
    # move "band below" up tho current band
    band_below_tray[topo] = ll
    # find and label connected components
    topo_con[:, :, ll], ncomponents[ll] = label(topo, structure)

# record band_below for each peak
peaklist.loc[:, "band_below"] = band_below_tray[peaks_xy]

print("Calculating prominence by band")
print("=============================")
col_elev_tray = elev.copy()  # record approximate elevation of col
# for each topo_con layer (top to bottom)
for ll in range(len(topo_elev) - 1, -1, -1):
    # find maximum within each neighborhood
    comp_peaks = maximum_position(elev, labels=topo_con[:, :, ll], index=list(range(1, ncomponents[ll] + 1)))
    # adjust estimation of col for current peaks
    col_elev_tray[tuple(np.array(list(zip(*comp_peaks))))] = topo_elev[ll]
    print(len(topo_elev) - ll, " of ", len(topo_elev))
# record col_elev for each peak
peaklist.loc[:, "col_elev"] = col_elev_tray[peaks_xy]
# estimate prominence by distance from peak height to col (assume col halfway between topo layers)
peaklist.loc[:, "prom_expected"] = peaklist.elev - (peaklist.col_elev - z_step/2)


print("Calculating isolation by band")
print("=============================")
# estimate isolation, begin at 0 (a bit slow...)
peaklist.loc[:, "iso_expected"] = 0
# for each elevation band
for ll in range(1, len(topo_elev)):
    # build tree of all points above current band (slow...)
    above_thresh = np.array(list(zip(*np.where(elev > topo_elev[ll]))))
    tree = spatial.KDTree(above_thresh, leafsize=10)
    # build query_list of all peaks between previous (lower) and current band
    query_list = np.array(list(zip(*[peaklist.peak_x[peaklist.band_below == ll-1], peaklist.peak_y[peaklist.band_below == ll-1]])))
    # calculate isolation as min distance from each peak to a higher point (dist in pixels)
    distance, index = tree.query(query_list)
    # record isolation in geo units
    peaklist.loc[peaklist.band_below == ll-1, "iso_expected"] = distance*ras.T0[0]
    print(ll, " of ", len(topo_elev)-1)

print("Writing peaks to file")
# calculate geo-coords
UTM_coords = ras.T1 * [peaklist.peak_y, peaklist.peak_x]
peaklist.loc[:, "UTM11N_x"] = UTM_coords[0]
peaklist.loc[:, "UTM11N_y"] = UTM_coords[1]

# write peaklist to file
output = peaklist.copy()
output = output.drop(["peak_x", "peak_y", "band_below", "col_elev"], axis=1)
output.to_csv(treetops_out, index=False)

# need to rewrite below using kdtrees function

# calculate distance from tree
parent_filtered = peaklist.copy()
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
rastools.raster_save(ras_nearest, nearest_out, data_format="int")

# export distance_map to raster file
ras_distance = ras
ras_distance.data = distance_map
rastools.raster_save(ras_distance, distance_out, data_format="float")

end = time.time()
print(end - start)

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# plt.imshow(band_below_tray, interpolation='nearest')