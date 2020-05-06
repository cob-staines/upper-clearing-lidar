import pandas as pd
import numpy as np
import hemigen

las_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\LAS\\19_149_all_200311_628000_5646525_vegetation.las"
hemi_out_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\opt\\os_0.25\\"

# filter hemisphere validation data
lookup_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\hemispheres\\hemi_lookup_cleaned.csv"
max_quality = 4
las_day = "19_149"
# import hemi_lookup
lookup = pd.read_csv(lookup_in)
# filter lookup
subset = lookup[lookup.quality_code <= max_quality]
# for ii in lasday
lookup_ss = subset[subset.folder == las_day]

# define parameters for hemigen
max_radius = 50  # in meters

optimization_scalar = 0.05
footprint = 0.15  # in m
c = 2834.64  # points to meters
point_size_scalar = footprint**2 * c * optimization_scalar

fig_size = 10  # in inches
fig_dpi = 100  # pixels/inch


# load_las
las_xyz = hemigen.las_load(las_in)

for ii in range(0, lookup_ss.shape[0]):
    # set point at ground level (x, y, z)
    ground_point = np.array(
        [lookup_ss.xcoordUTM1.iloc[ii], lookup_ss.ycoordUTM1.iloc[ii], lookup_ss.elevation.iloc[ii]])
    # correct for height offset of photography
    origin = ground_point + np.array([0, 0, lookup_ss.height_m.iloc[ii]])

    fig_out = hemi_out_dir + "las_" + las_day + "_img_" + lookup_ss.filename.iloc[ii][0:-4] + "_os_" + str(
        optimization_scalar) + ".png"

    hemigen.hemigen(las_xyz, origin, fig_out, max_radius, point_size_scalar, fig_size, fig_dpi)
