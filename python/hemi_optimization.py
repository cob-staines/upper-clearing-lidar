import pandas as pd
import numpy as np
import laslib

# poisson sample point cloud (las_in)
las_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\LAS\\19_149_snow_off_classified_merged.las"
min_radius = 0.15  # in meters
las_poisson_path = las_in.replace('.las', '_poisson_' + str(min_radius) + '.las')
# laslib.las_poisson_sample(las_in, min_radius, las_poisson_path)  # takes 20-30 minutes

# generate synthetic hemispheres
las_path = las_poisson_path
keep_class = 5
hdf5_path = las_path.replace('.las', '.hdf5')
# export las to hdf5
# laslib.las_xyz_to_hdf5(las_path, hdf5_path, keep_class=keep_class)

hemi_out_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\opt\\poisson\\"

# import hemi-photo lookup
lookup_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\hemispheres\\hemi_lookup_cleaned.csv"
max_quality = 4
las_day = "19_149"
# import hemi_lookup
lookup = pd.read_csv(lookup_in)
# filter lookup by quality
lookup = lookup[lookup.quality_code <= max_quality]
# filter lookup by las_day
lookup = lookup[lookup.folder == las_day]

# define parameters for hemigen
max_radius = 50  # in meters

optimization_scalar = 4
footprint = 0.15  # in m
c = 2834.64  # points to meters
point_size_scalar = footprint**2 * c * optimization_scalar

fig_size = 10  # in inches
fig_dpi = 100  # pixels/inch

origin_list = np.array([lookup.xcoordUTM1,
                       lookup.ycoordUTM1,
                       lookup.elevation + lookup.height_m]).swapaxes(0, 1)

fig_out_list = [hemi_out_dir + "las_" + las_day + "_img_" + fn[0:-4] + ".png" for fn in lookup.filename]

laslib.hemigen(hdf5_path, origin_list, fig_out_list, max_radius, point_size_scalar, fig_size, fig_dpi)
