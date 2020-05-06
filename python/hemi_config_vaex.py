import pandas as pd
# config

las_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\LAS\\19_149_all_200311_628000_5646525_vegetation.las"
lookup_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\hemispheres\\hemi_lookup_cleaned.csv"
hemi_out_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\opt\\os_0.25\\"
temp_file = las_in[0:-3] + 'hdf5'

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

import vaex

# import las file "las_in"
inFile = laspy.file.File(las_in, mode="r")
####
p0 = np.array([inFile.x,
               inFile.y,
               inFile.z,
               inFile.classification]).transpose()
classification = np.array(inFile.classification)

inFile.close()

class_filter = (classification != 7) & (classification != 8)

peace = vaex.from_arrays(x=p0[class_filter, 0], y=p0[class_filter, 1], z=p0[class_filter, 2])

# save to hdf5 for mem management
peace.export_hdf5(temp_file)

# clear mem
p0 = None
classification = None
peace = None

# pull from temp_file
vaex_df = vaex.open(temp_file)

# FOR ii IN LOOKUP_SS
# for ii in range(1, 2):
for ii in range(9, lookup_ss.shape[0]):

    # set point at ground level (x, y, z)
    ground_point = np.array([lookup_ss.xcoordUTM1.iloc[ii], lookup_ss.ycoordUTM1.iloc[ii], lookup_ss.elevation.iloc[ii]])
    # correct for height offset of photography
    origin = ground_point + np.array([0, 0, lookup_ss.height_m.iloc[ii]])

    # move to new origin
    vaex_df['x1'] = (vaex_df.x - origin[0]) * -1
    vaex_df['y1'] = vaex_df.y - origin[1]
    vaex_df['z1'] = vaex_df.z - origin[2]

    # calculate polar coords
    vaex_df['r'] = np.sqrt(vaex_df.x1**2 + vaex_df.y1**2 + vaex_df.z1**2)
    vaex_df['theta'] = np.arccos(vaex_df.z1 / vaex_df.r)
    vaex_df['phi'] = np.arctan2(vaex_df.y1, vaex_df.x1)
    vaex_df['area'] = ((footprint / vaex_df.r) ** 2) * c * optimization_scalar

    vaex_df['x2'] = vaex_df.theta * np.cos(vaex_df.phi)
    vaex_df['y2'] = vaex_df.theta * np.sin(vaex_df.phi)

    vaex_dff = vaex_df[vaex_df.r < max_radius]
    vaex_plot = vaex_dff['theta', 'phi', 'area']

    # to array
    array_plot = vaex_plot.to_arrays()

    fig_out = hemi_out_dir + "las_" + las_day + "_img_" + lookup_ss.filename.iloc[ii][0:-4] + "_os_" + str(
        optimization_scalar) + "_test.png"

    # matplotlib cleaner?

    fig = plt.figure(figsize=(figuresize, figuresize), dpi=fig_dpi, frameon=False)
    plt.scatter(array_plot[0], array_plot[1], s=array_plot[2], c='black')
    plt.savefig(fig_out)

    fig = plt.figure(figsize=(figuresize, figuresize), dpi=fig_dpi, frameon=False)
    ax = plt.axes([0., 0., 1., 1.], projection="polar", polar=True)
    sp1 = ax.scatter(array_plot[1], array_plot[0], s=array_plot[2], c="black")
    ax.set_rmax(np.pi / 2)
    ax.set_rticks([])
    ax.grid(False)
    ax.set_axis_off()
    fig.add_axes(ax)
    # Non-interactive.
    plt.savefig(fig_out)
    # Interactive.
    # plt.show()

    # parameters: figure size (match to images), point size scaling factor, thinning ratio

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