import pandas as pd

# config

las_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_all_test\\OUTPUT_FILES\\LAS\\19_149_all_test_628000_5646575_vegetation.las"
lookup_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\hemispheres\\hemi_lookup_cleaned.csv"
hemi_out_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\"

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
max_radius = 50  # in meters
sample_ratio = 1 / 5
footprint = 0.15
figuresize = 10

# HEMIGEN

import laspy
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

# import las_in
inFile = laspy.file.File(las_in, mode="r")
# select dimensions
p0 = np.array([inFile.x,
               inFile.y,
               inFile.z]).transpose()
classification = np.array(inFile.classification)
inFile.close()

# FOR ii IN LOOKUP_SS
for ii in range(0, 1):
    fig_out = hemi_out_dir + "las_" + las_day[0] + "_img_" + lookup_ss.filename.iloc[ii][0:-4] + ".png"

    # set point at ground level (x, y, z)
    ground_point = np.array([lookup_ss.xcoordUTM1.iloc[ii], lookup_ss.ycoordUTM1.iloc[ii], lookup_ss.elevation.iloc[ii]])
    # correct for height offset of photography
    origin = ground_point + np.array([0, 0, lookup_ss.height_m.iloc[ii]])

    # move to new origin
    p1 = p0 - origin

    # trivial subset (100m square around point, all points below horizon)
    subset_t = np.array([p1[:, 0] < max_radius,
                         p1[:, 0] > -max_radius,
                         p1[:, 1] < max_radius,
                         p1[:, 1] > -max_radius,
                         p1[:, 2] > 0,
                         classification != 7])

    p2 = p1[np.all(subset_t, 0), :]

    # calculate polar coords
    r = np.sqrt(np.sum(p2 ** 2, axis=1))
    # fine subset
    subset_f = r < max_radius

    r = r[subset_f]

    x = p2[subset_f, 0]
    y = p2[subset_f, 1]
    z = p2[subset_f, 2]

    # flip over x axis for upward-looking perspective
    x = -x

    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    # plotting

    # parameters: figure size (match to images), point size scaling factor, thinning ration

    # plot random subset of points
    sample_count = np.floor(phi.__len__() * sample_ratio).astype('int')
    test = random.sample(range(0, phi.__len__() - 1), sample_count)
    sort = np.flip(np.argsort(r[test]))

    # no color output
    c = 2834.64  # points to meters
    fig = plt.figure(figsize=(figuresize, figuresize))
    ax = fig.add_subplot(111, projection='polar')
    area = (footprint / r[test][sort]) ** 2
    c = ax.scatter(phi[test][sort], theta[test][sort], s=area * c, c="black")
    ax.set_rmax(np.pi / 2)
    ax.set_rticks([])
    ax.grid(False)

    fig.savefig(fig_out)
    print("done with " + fig_out)
