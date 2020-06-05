import rastools
import numpy as np
import pandas as pd
from skimage.morphology import reconstruction, opening
from scipy.ndimage.measurements import label, maximum_position
from sklearn.cluster import KMeans

# config
# raster chm for identifying treetops
ras_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\CHM\\19_149_snow_off_627975_5646450_spike_free_chm_.10m.bil"
# raster template for output nearest and distance maps
ras_map_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\CHM\\19_149_snow_off_627975_5646450_spike_free_chm_.10m.bil"
# output file naming conventions
output_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\DNT\\"
file_base = ras_in.split("\\")[-1].replace(".bil", "")
treetops_out = output_dir + file_base + "_kho_treetops.csv"

file_base = treetops_out.split("\\")[-1].replace("treetops.csv", "")
index_out = output_dir + file_base + "index_.10m.tif"
distance_out = output_dir + file_base + "distance_.10m.tif"

# parameters
z_min = 2
min_obj_rad_m = 1  # in meters
subpix_noise = True  # when true, peaks are randomly shifted at the subpixel scale (relative to CHM) to eliminate lattice effects in subsequent raster products

# load CHM
ras = rastools.raster_load(ras_in)

# define mask of valid data above  z_min
mask = (ras.data != ras.no_data) & (ras.data >= z_min)

# build opening structure
min_obj_rad_pix = min_obj_rad_m/ras.T0[0]

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

struct = mask_gen(min_obj_rad_pix)
chm = ras.data.copy()
chm[~mask] = 0

# morphologically open the CHM using the target minimum element struct
opened = opening(chm, struct)

# reconstruct the image using the opened image as the marker and the CHM as the mask
reconstructed = reconstruction(opened, chm)

# subtract the reconstructed image from the CHM to return all isolated elements above reconstructed image
neighborhoods = chm - reconstructed

# label neighborhoods
labels, nfeatures = label(neighborhoods, np.ones((3, 3)))

# calculate neighborhood area
area_pixels = np.bincount(labels.reshape([1, ras.rows*ras.cols])[0])[1:]

#identify regional maxima
regional_max = maximum_position(chm, labels=labels, index=list(range(1, nfeatures + 1)))


peak_xy = np.array(list(zip(*regional_max)))

peaklist = pd.DataFrame({"peak_x": peak_xy[0],
                         "peak_y": peak_xy[1],
                         "peak_z": ras.data[peak_xy[0], peak_xy[1]],
                         "area_pix": area_pixels,
                         "area_m2": area_pixels*(ras.T0[0]**2)})


kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(np.array(peaklist.area_m2).reshape(-1, 1))

peaklist.loc[:, "true_peak"] = kmeans.labels_

cluster_break = np.mean(kmeans.cluster_centers_)  # currently unused, could be recorded if useful

# add subpix noise
if subpix_noise:
    peaklist.peak_x = peaklist.peak_x + np.random.uniform(-0.5, 0.5, peaklist.__len__())
    peaklist.peak_y = peaklist.peak_y + np.random.uniform(-0.5, 0.5, peaklist.__len__())

print("Writing peaks to file")
# calculate geo-coords
UTM_coords = ras.T1 * [peaklist.peak_y, peaklist.peak_x]
peaklist.loc[:, "UTM11N_x"] = UTM_coords[0]
peaklist.loc[:, "UTM11N_y"] = UTM_coords[1]
peaklist.loc[:, "tree_id"] = peaklist.index

# write peaklist to file
output = peaklist.copy()
output = output.drop(["peak_x", "peak_y"], axis=1)
output.to_csv(treetops_out, index=False)

# reload peaklist if wishing to skip above calculations
# peaklist = pd.read_csv(treetops_out)

# filter to true peaks
peaks_filtered = peaklist.loc[peaklist.true_peak == 1, ['UTM11N_x', 'UTM11N_y']]
# load raster template for outputs
ras_map = rastools.raster_load(ras_map_in)

# calculate distance and index maps
index_map, distance_map = rastools.raster_nearest_neighbor(peaks_filtered, ras_map)

# export index_map to raster file
ras_index = ras_map
ras_index.data = index_map
rastools.raster_save(ras_index, index_out, data_format="uint32")

# export distance_map to raster file
ras_distance = ras_map
ras_distance.data = distance_map
rastools.raster_save(ras_distance, distance_out, data_format="float32")

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# plt.imshow(reconstructed, interpolation='nearest')
