import rastools
import numpy as np
import pandas as pd

# config
ras_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\CHM\\19_149_all_200311_628000_564652_chm_.10m.bil"

# output file naming conventions
output_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\DFT\\"
file_base = ras_in.split("\\")[-1].replace(".bil", "")
treetops_out = output_dir + file_base + "_pw_treetops.csv"
nearest_out = output_dir + file_base + "_pw_nearest.tif"
distance_out = output_dir + file_base + "_pw_distance_to_tree.tif"

# parameters
min_peak = 2

# Identifies treetops by finding local maximum pixel within variable window size (Popescu & Wynne 2004)
# Returns list of treetops, raster of distance to nearest treetop, and raster of domains for each treetop

# load CHM
ras = rastools.raster_load(ras_in)

def mask_size(height_m, unit_conversion=ras.T0[0]):
    # calculates window size (m) from Popescu & Wynne 2004
    len_m = 3.75105 - 0.17919 * height_m + 0.01241 * (height_m ** 2)
    # transform to pixels (round down)
    len_p = np.int(np.floor(len_m/unit_conversion))
    return len_p

def mask_gen(size):
    # generates circular mask of diameter mask_size (expected crown domain)
    # force odd size
    if (np.floor(size / 2) == size / 2):
        odd_size = size + 1
    else:
        odd_size = size
    mid = (odd_size - 1)/2
    n = odd_size
    r = size/2
    y, x = np.ogrid[-mid:n - mid, -mid:n - mid]
    mask = np.array(x * x + y * y <= r * r)
    return mask

# set edge buffer of max mask radius
buffer = np.ceil(mask_size(np.max(ras.data))/2)

# preallocate
peaklist = []
# for each pixel
for ii in range(buffer, ras.rows-buffer):
    for jj in range(buffer, ras.cols-buffer):
        # if at or above min_peak
        if ras.data[ii, jj] >= min_peak:
            # get mask dimensions
            mask_len = mask_size(ras.data[ii, jj])
            # get mask
            mask = mask_gen(mask_len)
            # get radius
            offset = np.int((np.shape(mask)[0]-1)/2)
            # pull local sample from ras
            bite = ras.data[(ii-offset):(ii+offset+1), (jj-offset):(jj+offset+1)]
            # mask
            max_bite_mask = max(bite[mask])
            # if local maximum
            if max_bite_mask == ras.data[ii, jj]:
                # mark as peak
                peaklist.append((ii, jj))

# calculate coordinates of peak pixel centers
peak_ij = list(zip(*peaklist))
# convert to data frame
peak_df = pd.DataFrame({'index_ii': peak_ij[0],
                        'index_jj': peak_ij[1],
                        'height_m': ras.data[np.array(peak_ij[0]), np.array(peak_ij[1])]})

UTM_coords = ras.T1 * [peak_df.index_jj, peak_df.index_ii]
peak_df.loc[:, "UTM11N_x"] = UTM_coords[0]
peak_df.loc[:, "UTM11N_y"] = UTM_coords[1]

# write peak_df to file
output = peak_df.drop(["index_ii", "index_jj"], axis=1)
output = output.reset_index()
output.to_csv(treetops_out, index=False)

# distance to tree map (brute force, 75s at .25m res)
# preallocate
nearest_map = np.full_like(ras.data, ras.no_data)
distance_map = np.full_like(ras.data, np.nan)
# for each pixel
for ii in range(0, ras.cols):
    for jj in range(0, ras.rows):
        # find cell coords
        cell_coords = ras.T1 * [ii, jj]
        # find distances to all peaks
        distances = np.sqrt((cell_coords[0] - np.array(peak_df.UTM11N_x))**2 + (cell_coords[1] - np.array(peak_df.UTM11N_y))**2)
        # find and record index of minimum
        nearest_id = np.argmin(distances)
        nearest_map[jj, ii] = nearest_id
        # record minimum
        distance_map[jj, ii] = distances[nearest_id]

# export parent_map to raster file
ras_nearest = ras
ras_nearest.data = nearest_map
rastools.raster_save(ras_nearest, nearest_out, data_format="int")

# export distance_map to raster file
ras_distance = ras
ras_distance.data = distance_map
rastools.raster_save(ras_distance, distance_out, data_format="float")
