import laspy
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random

workingdir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_all_test\\"
las_in = workingdir + "OUTPUT_FILES\\LAS\\19_149_all_test_628000_5646575_vegetation.las"

# import las_in
inFile = laspy.file.File(las_in, mode="r")
# select dimensions
p0 = np.array([inFile.x,
               inFile.y,
               inFile.z]).transpose()
classification = np.array(inFile.classification)
inFile.close()

max_radius = 50  # in meters

# FOR EACH SAMPLE POINT:

fig_out = workingdir + "false_hemis\\19_149_DSCN6393.png"

# set point (at ground level)
ground_point = np.array([628118.845606028568000, 5646597.918613061308861, 1830.321999999999889])
# correct for 1.8m offset of photography
origin = ground_point + np.array([0, 0, 1.8])

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

theta = np.arccos(z/r)
phi = np.arctan2(y, x)

# plotting

# parameters: figure size (match to images), point size scaling factor, thinning ration

# plot random subset of points
sample_ratio = 1/10
sample_count = np.floor(phi.__len__()*sample_ratio).astype('int')
test = random.sample(range(0, phi.__len__()-1), sample_count)
sort = np.flip(np.argsort(r[test]))

# no color
footprint = 0.15
c = 2834.64  # points to meters
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='polar')
area = (footprint/r[test][sort])**2
c = ax.scatter(phi[test][sort], theta[test][sort], s=area*c, c="black")
ax.set_rmax(np.pi/2)
ax.set_rticks([])
ax.grid(False)

# color
footprint = 0.15
c = 2834.64  # points to meters
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='polar')
area = (footprint/r[test][sort])**2
c = ax.scatter(phi[test][sort], theta[test][sort], s=area*c, c=r[test][sort], cmap='hsv')
ax.set_rmax(np.pi/2)
ax.set_rticks([])
ax.grid(False)
fig.colorbar(c, ax=ax)

fig.savefig(fig_out)


# plot all points
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='polar')
c = ax.scatter(phi, theta, s=20/r, c=r, cmap='hsv')
ax.set_rmax(np.pi/2)
ax.get_thetaaxis()

fig.savefig(fig_out)
