import laspy
import pandas as pd
import matplotlib
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
las_time = pd.DataFrame({'Time[s]': inFile.gps_time})

# import trajectory
traj = pd.read_csv(filedir + traj_in)
# throw our pitch, roll, yaw (or until needed later)
traj = traj[['Time[s]', 'Easting[m]', 'Northing[m]', 'Height[m]']]

# resample traj to las gps times and interpolate

# append las times to traj
outer = traj.append(las_time, sort=0)
# order by gps time
outer = outer.sort_values(by="Time[s]")
# set index as gps_time for nearest neighbor interpolation
outer = outer.set_index('Time[s]')
# interpolate by nearest neighbor
interpolated = outer.interpolate(method="nearest")
interpolated = interpolated.reset_index()

output_file = laspy.file.File(filedir + traj_out, mode="w", header=inFile.header)
output_file.gps_time = interpolated['Time[s]'].values
output_file.x = interpolated['Easting[m]'].values
output_file.y = interpolated['Northing[m]'].values
output_file.z = interpolated['Height[m]'].values
output_file.close()

# compare points with traj
points = pd.DataFrame({'Time[s]': inFile.gps_time, 'x': inFile.x, 'y': inFile.y, 'z': inFile.z})
comparison = interpolated.merge(points, on="Time[s]", how='inner')

