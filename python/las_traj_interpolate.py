import laspy
import pandas as pd
import matplotlib
import numpy as np
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

# config
filedir = """C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\LiDAR\\19_050\\"""
las_in = """OUTPUT_FILES\\19_050_ladder_clearing_WGS84_utm11N_clearing_ground-points_1st-return_5deg.las"""
traj_in = '19_050_all_WGS84_utm11N_trajectories.txt'
traj_out = '19_050_all_WGS84_utm11N_trajectories_interpolated_clearing_ladder.las'

# import las
inFile = laspy.file.File(filedir + las_in, mode="r")
# pull only gps_time
las_data = pd.DataFrame({'gps_time': inFile.gps_time,
                         'x': inFile.x,
                         'y': inFile.y,
                         'z': inFile.z,
                         'intensity': inFile.intensity})

# import trajectory
traj = pd.read_csv(filedir + traj_in)
# rename columns for consistency
traj = traj.rename(columns={'Time[s]': "gps_time",
                            'Easting[m]': "easting_m",
                            'Northing[m]': "northing_m",
                            'Height[m]': "height_m"})
# throw our pitch, roll, yaw (at least until needed later...)
traj = traj[['gps_time', 'easting_m', 'northing_m', 'height_m']]

# resample traj to las gps times and interpolate

# outer merge las and traj on gps_time
outer = traj.merge(las_data.gps_time, on="gps_time", how="outer")

# order by gps time
outer = outer.sort_values(by="gps_time")
# set index as gps_time for nearest neighbor interpolation
outer = outer.set_index('gps_time')
# interpolate by nearest neighbor
interpolated = outer.interpolate(method="nearest") #issues with other columns.... can we specify?
# resent index for clarity
interpolated = interpolated.reset_index()

# calculate point distance from track
las_data = las_data.merge(interpolated, on="gps_time", how="left")
p1 = np.array([las_data.easting_m, las_data.northing_m, las_data.height_m])
p2 = np.array([las_data.x, las_data.y, las_data.z])
squared_dist = np.sum((p1-p2)**2, axis=0)
las_data = las_data.assign(distance_to_track=np.sqrt(squared_dist))

# plot height vs. distance
las_data.plot.scatter(x='distance_to_track', y='intensity')

# linear regression
