import laspy
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

# config
filedir = """C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\LiDAR\\19_149\\"""
las_in = """19_149_ladder_forest_WGS84_utm11N_nocolor.las"""
traj_in = '19_149_trajectory_all.txt'
traj_out = '19_149_trajectory_interpolated_forest_ladder.las'


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
outer = outer.set_index('index')
# interpolate by nearest neighbor
interpolated = outer.interpolate(method="nearest")

peace = inFile
peace.gps_time = interpolated['Time[s]'].values
peace.x = interpolated['Easting[m]'].values
peace.y = interpolated['Northing[m]'].values
peace.z = interpolated['Height[m]'].values

output_file = laspy.file.File(filedir + traj_out, mode="w", header=inFile.header)
output_file.gps_time = interpolated['Time[s]'].values
output_file.x = interpolated['Easting[m]'].values
output_file.y = interpolated['Northing[m]'].values
output_file.z = interpolated['Height[m]'].values
output_file.close()

