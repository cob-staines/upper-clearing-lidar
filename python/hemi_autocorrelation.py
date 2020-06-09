import pandas as pd
import numpy as np
import laslib
import rastools
import os

import time


# pull ground elevation of sample points from 19_149_dem_.5m
ras_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\DEM\\19_149_all_200311_628000_5646525dem_.50m.bil"
pts_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\site_library\\upper-forest_autocorrelation_sample_points.csv"
pts_out = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\site_library\\upper-forest_autocorrelation_sample_points_elev.csv"
rastools.point_sample_raster(ras_in, pts_in, pts_out, "x_utm11n", "y_utm11n", "elev_m", "NA")

# generate synthetic hemispheres
las_path = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\LAS\\19_149_snow_off_clean.las"
hdf5_path = las_path[0:-3] + 'hdf5'
hemi_out_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\autocorrelation\\uf\\"

# import lookup
lookup_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\site_library\\upper-forest_autocorrelation_sample_points_elev.csv"
lookup = pd.read_csv(lookup_in)
hemi_height = 1.8  #m above ground

# define parameters for hemigen
max_radius = 50  # in meters

optimization_scalar = 0.05
footprint = 0.15  # in m
c = 2834.64  # points to meters
point_size_scalar = footprint**2 * c * optimization_scalar

fig_size = 10  # in inches
fig_dpi = 100  # pixels/inch

# export las to hdf5
laslib.las_xyz_to_hdf5(las_in, las_hdf5)

for ii in range(39, lookup.shape[0]):
    # set point at ground level (x, y, z)
    origin = np.array([lookup.x_utm11n[ii], lookup.y_utm11n[ii], lookup.elev_m[ii] + hemi_height])

    fig_out = hemi_out_dir + "las_19_149_pnt_" + lookup.id[ii].astype(str) + ".png"

    start = time.time()
    laslib.hemigen(las_hdf5, origin, fig_out, max_radius, point_size_scalar, fig_size, fig_dpi)
    end = time.time()
    print(end - start)

# remove temp file
os.remove(las_hdf5)
