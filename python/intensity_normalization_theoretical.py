
#dependencies

import laspy
import pandas as pd
import matplotlib
import numpy as np
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from las_traj import las_traj

# config
# note that laspy only works with las files, laz (in or out) will produce an error
filedir = """C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\LiDAR\\19_045\\"""
# las_norm_in points to the las file to be range normalized
las_norm_in = """19_045_vertical_clearing_02_intensity-analysis\\OUTPUT_FILES\\19_045_vertical_clearing_02_intensity-analysis_clearing_ground-points_single-return_5deg.las"""
#las_norm_out points to the file when the normalized output will be saved
las_norm_out = """19_045_vertical_clearing_02_intensity-analysis\\OUTPUT_FILES\\19_045_vertical_clearing_02_intensity-analysis_clearing_ground-points_single-return_5deg_range-norm.las"""
# traj_in points to the trajectory file (must be common to both las_ref_in and las_norm_in
traj_in = '19_045_all_trajectories_WGS84_utm11N.txt'

# calculate distance and angle
las_in = filedir + las_norm_in
traj_in = filedir + traj_in

las_data = las_traj(filedir + las_norm_in, filedir + traj_in)

# theoretical correction

norm_range = 80
# intensity is decibel scale, need to back-calculate to get proportional to power
intensity_proportional = 10 ** (las_data.intensity/1000)
# correct for intensity ~ distance^-2
las_data = las_data.assign(intensity_range_norm=(intensity_proportional * las_data.distance_to_track ** 2)/norm_range ** 2)

# visually inspect with plots if desired
# plt.scatter(las_data.distance_to_track, las_data.intensity_range_norm, s=1)
# plt.scatter(las_data.angle_from_nadir_deg, las_data.intensity_range_norm, s=1)


# write output to file (commit to another function)
inFile = laspy.file.File(filedir + las_norm_in, mode="r")
outFile = laspy.file.File(filedir + las_norm_out, mode="w", header=inFile.header)

# define new dimensions
outFile.define_new_dimension(name="angle_from_nadir", data_type=3, description="angle in degrees from nadir")

# pull all dimensions from input file
for dimension in inFile.point_format:
    dat = inFile.reader.get_dimension(dimension.name)
    outFile.writer.set_dimension(dimension.name, dat)

# close input file
inFile.close()

# save new dimensions
outFile.intensity = las_data.intensity_range_norm
outFile.angle_from_nadir = las_data.angle_from_nadir_deg

# close output file
outFile.close()
