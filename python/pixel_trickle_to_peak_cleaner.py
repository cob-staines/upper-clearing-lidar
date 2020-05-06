import numpy as np
import pandas as pd
import rastools

# config
ras_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\CHM\\19_149_all_200311_628000_564652_chm_.25m.bil"

# output file naming conventions
output_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\DFT\\"
file_base = ras_in.split("\\")[-1].replace(".bil", "")
treetops_out = output_dir + file_base + "_trickle_treetops.csv"
parent_out = output_dir + file_base + "_trickle_parent.tif"
nearest_out = output_dir + file_base + "_trickle_nearest.tif"
distance_out = output_dir + file_base + "_trickle_distance_to_tree.tif"

# parameters
min_vegetation_height = 2  # in meters
min_radius = 1  # in meters



# load raster
ras = rastools.raster_load(ras_in)

# calculate min_pixel threshold
pixel_area = np.array(ras.T1*(0, 0)) - np.array(ras.T1*(1, 1))
pixel_area = pixel_area[1] ** 2

min_area = np.pi*min_radius ** 2
min_pixel = int(min_area/pixel_area)

# trickle up to peak
seed = ras.data > min_vegetation_height
peakmap = np.full([ras.rows, ras.cols, 2], np.nan)

# for each pixel
for ii in range(1, ras.rows-2):
    for jj in range(1, ras.cols-2):
        x = ii
        y = jj
        if seed[ii, jj]:
            moved = True
            while moved:
                window = ras.data[x-1:x+2, y-1:y+2]
                dx, dy = np.subtract(np.unravel_index(window.argmax(), window.shape), (1, 1))
                x = x + dx
                y = y + dy
                if (dx == 0) & (dy == 0):
                    moved = False
                    peakmap[ii, jj, :] = (x, y)
                if (x == 0) or (x == ras.rows-1) or (y == 0) or (y == ras.cols-1):
                    moved = False

# list of peak by pixel
peaklist = pd.DataFrame({'peak_x': peakmap[:, :, 0].flatten(),
                         'peak_y': peakmap[:, :, 1].flatten()})
# list of unique peaks
peakcount = peaklist.groupby(['peak_x', 'peak_y']).size().reset_index().rename(columns={0: 'area'})

# threshold by mix_pixel (area)
peak_over_thresh = peakcount[peakcount.area >= min_pixel].copy()

# merge index with map to visualize crowns captured
# assign index
peak_over_thresh = peak_over_thresh.reset_index(drop=True)
peak_over_thresh.loc[:, "peak_index"] = peak_over_thresh.index

# output peaklist
# calculate geo-coords
UTM_coords = ras.T1 * [peak_over_thresh.peak_y, peak_over_thresh.peak_x]
peak_over_thresh.loc[:, "UTM11N_x"] = UTM_coords[0]
peak_over_thresh.loc[:, "UTM11N_y"] = UTM_coords[1]
peak_over_thresh.loc[:, "area_m"] = peak_over_thresh.area*pixel_area

# write parentlist to file
output = peak_over_thresh.copy()
output = output.drop(["peak_x", "peak_y", "area"], axis=1)
output.to_csv(treetops_out, index=False)

# output domains
remap = pd.merge(peaklist, peak_over_thresh, how="left", on=("peak_x", "peak_y"))
parent_map = np.array(remap.peak_index).reshape((ras.rows, ras.cols))
ras_parent = ras
ras_parent.data = parent_map
rastools.raster_save(ras_parent, parent_out, data_format="int")

# output distance from peak
# preallocate
nearest_map = np.full_like(ras.data, ras.no_data)
distance_map = np.full_like(ras.data, np.nan)
# slow but works (60s?)
for ii in range(0, ras.cols):
    for jj in range(0, ras.rows):
        cell_coords = ras.T1 * [ii, jj]
        distances = np.sqrt((cell_coords[0] - np.array(peak_over_thresh.UTM11N_x))**2 + (cell_coords[1] - np.array(peak_over_thresh.UTM11N_y))**2)
        nearest_id = np.argmin(distances)
        nearest_map[jj, ii] = peak_over_thresh.peak_index[nearest_id]
        distance_map[jj, ii] = distances[nearest_id]

ras_nearest = ras
ras_nearest.data = nearest_map
rastools.raster_save(ras_nearest, nearest_out, data_format="int")

ras_distance = ras
ras_distance.data = distance_map
rastools.raster_save(ras_distance, distance_out, data_format="int")


# import matplotlib.pyplot as plt
# plt.imshow(parent_map, interpolation='nearest')
