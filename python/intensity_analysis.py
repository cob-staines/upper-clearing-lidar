import laspy
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

# config
filedir = """C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\LiDAR\\19_050\\OUTPUT_FILES\\"""
las_in = """19_050_ladder_clearing_WGS84_utm11N_clearing_ground-points_1st-return_5deg_track.las"""

# import las
inFile = laspy.file.File(filedir + las_in, mode="r")

setattr(inFile, 'xyz_range_from_track', getattr(inFile, 'xyz-range_from_track'))

# pull only gps_time
peace = pd.DataFrame({'intensity': inFile.intensity, 'distance': inFile.xyz_range_from_track})
peace.plot.scatter(x='distance', y='intensity')
