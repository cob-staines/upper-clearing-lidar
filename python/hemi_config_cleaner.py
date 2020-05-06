import pandas as pd
# config

las_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\LAS\\19_149_all_200311_628000_5646525_vegetation.las"
lookup_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\hemispheres\\hemi_lookup_cleaned.csv"
hemi_out_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\opt\\os_0.25\\"

# filter hemisphere validation data
max_quality = 4
las_day = "19_149"

# import hemi_lookup
lookup = pd.read_csv(lookup_in)
# filter lookup
subset = lookup[lookup.quality_code <= max_quality]
# for ii in lasday
lookup_ss = subset[subset.folder == las_day]

# synthetic hemisphere parameters
optimization_scalar = 0.05
max_radius = 50  # in meters
footprint = 0.15  # in m
c = 2834.64  # points to meters
figuresize = 10  # in inches
fig_dpi = 100  # pixels/inch

# HEMIGEN

import laspy
import numpy as np
import matplotlib

matplotlib.use('Agg')
# matplotlib.use('TkAgg')  # use for interactive plotting
import matplotlib.pyplot as plt
import random

# load_las

# import las file "las_in"
inFile = laspy.file.File(las_in, mode="r")
####
p0 = np.array([inFile.x,
               inFile.y,
               inFile.z]).transpose()
classification = np.array(inFile.classification)

inFile.close()

class_filter = (classification != 7) & (classification != 8)
classification = None

# remove noise
p0 = p0[class_filter]

# hemigen
# inputs: las var (filtered), origin coords, prarams, file locations)

# FOR ii IN LOOKUP_SS
for ii in range(0, lookup_ss.shape[0]):

    # set point at ground level (x, y, z)
    ground_point = np.array([lookup_ss.xcoordUTM1.iloc[ii], lookup_ss.ycoordUTM1.iloc[ii], lookup_ss.elevation.iloc[ii]])
    # correct for height offset of photography
    origin = ground_point + np.array([0, 0, lookup_ss.height_m.iloc[ii]])

    # move to new origin
    p1 = p0 - origin

    # less memory:
    r = np.sqrt(np.sum(p1 ** 2, axis=1))
    subset_f = r < max_radius

    r = r[subset_f]
    p1 = p1[subset_f]

    # flip over x axis for upward-looking perspective
    p1[:, 0] = -p1[:, 0]

    # calculate polar coords
    data = pd.DataFrame({'theta': np.arccos(p1[:, 2] / r),
                         'phi': np.arctan2(p1[:, 1], p1[:, 0]),
                         'area': ((footprint / r) ** 2) * c * optimization_scalar})

    # mem management
    p1 = None

    # plot

    fig_out = hemi_out_dir + "las_" + las_day + "_img_" + lookup_ss.filename.iloc[ii][0:-4] + "_os_" + str(
        optimization_scalar) + ".png"

    fig = plt.figure(figsize=(figuresize, figuresize), dpi=fig_dpi, frameon=False)
    ax = plt.axes([0., 0., 1., 1.], projection="polar", polar=True)
    sp1 = ax.scatter(data.phi, data.theta, s=data.area, c="black")
    ax.set_rmax(np.pi / 2)
    ax.set_rticks([])
    ax.grid(False)
    ax.set_axis_off()
    fig.add_axes(ax)

    fig.savefig(fig_out)
    print("done with " + fig_out)
