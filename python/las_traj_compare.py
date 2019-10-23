import laspy
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

filedir = """C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\LiDAR\\19_149\\"""
filename_in = """19_149_ladder_forest_WGS84_utm11N_nocolor - Scanner 1 - 190529_184952_Scanner_1 - originalpoints.las"""
datapath = filedir + filename_in

# read data in
inFile = laspy.file.File(datapath, mode="r")
las_time = pd.Series(inFile.gps_time, name="Time[s]")
sub = pd.Series(las_time.nlargest(1000))


traj = pd.read_csv(filedir + '19_149_trajectory_all.txt')

output = traj.join(sub, how='right', lsuffix='_left', rsuffix='_right')
