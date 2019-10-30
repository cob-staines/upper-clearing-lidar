import laspy
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# config
filedir = """C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\LiDAR\\19_149\\"""
datapath = filedir + """19_149_ladder_forest_WGS84_utm11N_nocolor.las"""

# read data in
inFile = laspy.file.File(datapath, mode="r")
# isolate gps_time
las_time = pd.DataFrame({'Time[s]': inFile.gps_time})

#subset for test case
sub = las_time[:1000]

#import trajectory
traj = pd.read_csv(filedir + '19_149_trajectory_all.txt')

#join trajectory gps time with subset of las gps times
inner = traj.merge(sub, on="Time[s]", how='inner')
# no overlap found between subset of las gps times and trajectory file.

# resample traj to las gps times and linearly interpolate

outer = traj.merge(sub, on="Time[s]", how='outer')
# las times with no matching traj time are appended to bottom, order by time to see spacing
outer = outer.sort_values(by="Time[s]")
#interpolate to fill nans
outer = outer.set_index('Time[s]')
interpolated = outer.interpolate(method="nearest")
