
#dependencies

import laspy
import pandas as pd
import matplotlib
import numpy as np
matplotlib.use('TkAgg')
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from las_traj import las_traj

# config
# note that laspy only works with las files, laz (in or out) will produce an error
filedir = """C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\LiDAR\\19_045\\"""
# las_ref_in points to the las-file used for curve-fitting in range normalization
las_ref_in = """19_045_vertical_clearing_02_intensity-analysis\\OUTPUT_FILES\\19_045_vertical_clearing_02_intensity-analysis_clearing_ground-points_single-return_5deg.las """
# las_norm_in points to the las file to be range normalized
las_norm_in = """19_045_ladder_clearing_intensity-analysis\\OUTPUT_FILES\\19_045_ladder_clearing_intensity-analysis_vegetation-points_above-15m.las"""
#las_norm_out points to the file when the normalized output will be saved
las_norm_out = """19_045_ladder_clearing_intensity-analysis\\OUTPUT_FILES\\19_045_ladder_clearing_intensity-analysis_vegetation-points_above-15m.las_range-normalized.las"""
# traj_in points to the trajectory file (must be common to both las_ref_in and las_norm_in
traj_in = '19_045_all_trajectories_WGS84_utm11N.txt'

# calculate distance and angle
las_data, ref_header = las_traj(filedir + las_ref_in, filedir + traj_in)

# bin and quantile

# curve_x defined by bin midpoints
bin_count = 50
bins = np.linspace(las_data.distance_to_track.min(), las_data.distance_to_track.max(), bin_count)
curve_x = bins[:-1] + (bins[1:] - bins[:-1]) / 2

# curve_y defined by 95th percentile of bin to mode highest intensities
bin_groups = las_data.intensity.groupby(np.digitize(las_data.distance_to_track, bins))
curve_y = bin_groups.quantile(.95)[:-1]


# define cure-fitting function
def func(x, a, b, c):
    return a * np.exp(-b * x) * x ** -c


# fit curve
popt, pcov = curve_fit(func, curve_x, curve_y, p0=[1000., .001, 1])

# plot to check fit visually
plt.scatter(las_data.distance_to_track, las_data.intensity, s=1)
plt.scatter(curve_x, curve_y)
plt.plot(curve_x, func(curve_x, *popt), color='red')

# memory management
las_data = None
ref_header = None

# normalize las_norm_in intensity using fitted curve
las_norm_data, norm_header = las_traj(filedir + las_norm_in, filedir + traj_in)


def func(I, x1, x2, b, c):
    return I * np.exp(b * (x1 - x2)) * x1 ** c * x2 ** -c


b = popt[1]
c = popt[2]
norm_distance = 80
temp = func(las_norm_data.intensity, las_norm_data.distance_to_track, norm_distance, b, c)
las_norm_data = las_norm_data.assign(intensity_range_norm=temp)

# inspect corrected data manually if desired
plt.scatter(las_norm_data.distance_to_track, las_norm_data.intensity_range_norm, s=1)

# write output to file
inFile = laspy.file.File(filedir + las_norm_in, mode="r")
datapath = filedir + las_norm_out
output_file = laspy.file.File(datapath, mode="w", header=inFile.header)
inFile.close()
# does this handle all the data correctly? lets do a comparison

output_file.gps_time = las_norm_data.gps_time
output_file.x = las_norm_data.x
output_file.y = las_norm_data.y
output_file.z = las_norm_data.z
# need to pass classification....

output_file.intensity = las_norm_data.intensity_range_norm
#output_file.scan_angle = las_norm_data.angle_from_nadir_deg
output_file.close()

