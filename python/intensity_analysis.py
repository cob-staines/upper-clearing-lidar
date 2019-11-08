import laspy
import pandas as pd
import matplotlib
import numpy as np
matplotlib.use('TkAgg')
from sklearn import linear_model

import matplotlib.pyplot as plt


# config
filedir = """C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\LiDAR\\19_050\\"""
las_in = """OUTPUT_FILES\\19_050_ladder_clearing_WGS84_utm11N_clearing_ground-points_single-return_5deg.las"""
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
inFile.close()

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

#range normalization (Okhrimernko and Hopkinson 2019)
norm_range=60 #m
las_data = las_data.assign(intensity_norm=las_data.intensity*las_data.distance_to_track**2/(norm_range**2))
#atmospheric losses not constant

plt02 = las_data.plot.scatter(x='distance_to_track', y='intensity')
plt01.set_ylim(0, 5500)
plt01.set_xlim(0, 120)

plt01 = las_data.plot.scatter(x='distance_to_track', y='intensity_norm')
plt01.set_ylim(0, 5500)
plt01.set_xlim(0, 120)

# linear regression
# create regression object

# get x and y vectors

# calculate polynomial
z = np.polyfit(las_data.distance_to_track, las_data.intensity_norm, 2)
f = np.poly1d(z)

# calculate new x's and y's
x_new = np.linspace(las_data.distance_to_track.min(), las_data.distance_to_track.max(), 50)
y_new = f(x_new)

ax = plt.gca()

plt02 = las_data.plot.scatter(x='distance_to_track', y='intensity_norm', ax=ax)
plt02 = plt.plot(x_new, y_new)

plt.show()

plt02.ylim([0, 5500])
plt02.set_xlim(0, 120)


regr = linear_model.LinearRegression()
regr.fit(las_data.distance_to_track.values.reshape((-1, 1)), las_data.intensity.values)
print(regr.intercept_)
print(regr.coef_)
