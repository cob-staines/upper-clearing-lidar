import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd



# config
ras_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_all_test\\OUTPUT_FILES\\CHM\\19_149_all_test_628000_564657pit_free_chm_.25m.bil"
las_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_vertical_forest_WGS84_utm11N.las"
traj_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_all_traj.txt"


from raster_load import raster_load


# this takes a while
from las_traj import las_traj


def Bresenham3D(x1, y1, z1, x2, y2, z2):
    ListOfPoints = []
    ListOfPoints.append((x1, y1, z1))
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)
    if (x2 > x1):
        xs = 1
    else:
        xs = -1
    if (y2 > y1):
        ys = 1
    else:
        ys = -1
    if (z2 > z1):
        zs = 1
    else:
        zs = -1

    # Driving axis is X-axis"
    if (dx >= dy and dx >= dz):
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while (x1 != x2):
            x1 += xs
            if (p1 >= 0):
                y1 += ys
                p1 -= 2 * dx
            if (p2 >= 0):
                z1 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            ListOfPoints.append((x1, y1, z1))

            # Driving axis is Y-axis"
    elif (dy >= dx and dy >= dz):
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while (y1 != y2):
            y1 += ys
            if (p1 >= 0):
                x1 += xs
                p1 -= 2 * dy
            if (p2 >= 0):
                z1 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            ListOfPoints.append((x1, y1, z1))

            # Driving axis is Z-axis"
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while (z1 != z2):
            z1 += zs
            if (p1 >= 0):
                y1 += ys
                p1 -= 2 * dz
            if (p2 >= 0):
                x1 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
            ListOfPoints.append((x1, y1, z1))
    return ListOfPoints


def main():
    (x1, y1, z1) = (-100, 1, 1)
    (x2, y2, z2) = (5, 3, -1)
    ListOfPoints = Bresenham3D(x1, y1, z1, x2, y2, z2)
    print(ListOfPoints)

ras = raster_load(ras_in)
traj = las_traj(las_in, traj_in)

# pull xy resolution
res = (ras.T0 * (1, 0))[0] - (ras.T0 * [0, 0])[0]

# convert to affine
sink_xy = np.rint(np.array(~ras.T0 * [traj.x, traj.y]))
source_xy = np.rint(np.array(~ras.T0 * [traj.easting_m, traj.northing_m]))

# why are negative values appearing here? because point cloud has not been cropped to area... should not be an issue if using processed las file
mask = (sink_xy[0] >= 0) & (sink_xy[0] < ras.rows) & (sink_xy[1] >= 0) & (sink_xy[1] < ras.cols)
mask = mask & (source_xy[0] >= 0) & (source_xy[0] < ras.rows) & (source_xy[1] >= 0) & (source_xy[1] < ras.cols)

# elevation affine conversion (use same resolution as xy)
# h_a = (h - height_cutoff_low)/res
# h = res*h_a + height_cutoff_low
height_cutoff_low = 1800  # height of cell 0
height_cutoff_high = 1880

# mask any below or above height thresholds
mask = mask & (traj.z >= height_cutoff_low) & (traj.z < height_cutoff_high)

sink_h_a = np.rint((traj.z - height_cutoff_low)/res)
source_h_a = np.rint((traj.height_m - height_cutoff_low)/res)

sink = np.array([sink_xy[0], sink_xy[1], sink_h_a])
sink = sink[:, mask]

# clear memory

sink_xy = None
sink_h_a = None
source_h_a = None
source_xy = None

source = np.array([source_xy[0], source_xy[1], source_h_a])
source = source[:, mask]

# preallocate
levels = int(np.max(source[2, :])) + 1

traversals = np.full([ras.rows, ras.cols, levels], 0)
terminals = np.full([ras.rows, ras.cols, levels], 0)

# subset
sublength = 10
sub = np.random.permutation(np.shape(sink)[1])[0:sublength]

sink_s = sink[:, sub]
source_s = source[:, sub]

for ii in range(0, sublength):
    pointlist = Bresenham3D(sink_s[0, ii], sink_s[1, ii], sink_s[2, ii], source_s[0, ii], source_s[1, ii], source_s[2, ii])
    fist = list(zip(*pointlist))

# this may be pointless


main()