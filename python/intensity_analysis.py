import laspy
import pandas as pd
import matplotlib
import numpy as np

matplotlib.use('TkAgg')
from sklearn import linear_model
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

# config
filedir = """C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\LiDAR\\19_045\\"""
las_in = """19_045_vertical_clearing_02_intensity-analysis\\OUTPUT_FILES\\19_045_vertical_clearing_02_intensity-analysis_clearing_ground-points_single-return_5deg.las"""
las_out = """19_045_ladder_clearing_intensity-analysis\\OUTPUT_FILES\\19_045_ladder_clearing_intensity-analysis_clearing_ground-points_single-return_5deg_range-normalized.las"""
traj_in = '19_045_all_trajectories_WGS84_utm11N.txt'

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
interpolated = outer.interpolate(method="nearest")  # issues with other columns.... can we specify?
# resent index for clarity
interpolated = interpolated.reset_index()

# calculate point distance from track
las_data = las_data.merge(interpolated, on="gps_time", how="left")
p1 = np.array([las_data.easting_m, las_data.northing_m, las_data.height_m])
p2 = np.array([las_data.x, las_data.y, las_data.z])
squared_dist = np.sum((p1 - p2) ** 2, axis=0)
las_data = las_data.assign(distance_to_track=np.sqrt(squared_dist))

# range normalization (Okhrimernko and Hopkinson 2019)
norm_range = 80  # m
las_data = las_data.assign(intensity_norm=las_data.intensity * las_data.distance_to_track ** 2 / norm_range ** 2)
# atmospheric losses not constant

las_data = las_data.assign(intensity_log=np.log(las_data.intensity))

# curve-fitting
bins = np.linspace(las_data.distance_to_track.min(), las_data.distance_to_track.max(), 45)
peace = las_data.intensity.groupby(np.digitize(las_data.distance_to_track, bins))

train = peace.quantile(.95)
train = train[:14]
bins_mp = bins[:-1] + (bins[1:] - bins[:-1]) / 2

plt.plot(bins_mp, train)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(bins_mp, peace.quantile(.95)[:-1])
plt.plot(bins_mp, peace.quantile(.9)[:-1])
plt.plot(bins_mp, peace.quantile(.8)[:-1])
plt.plot(bins_mp, peace.quantile(.7)[:-1])
plt.plot(bins_mp, peace.quantile(.6)[:-1])
plt.plot(bins_mp, peace.quantile(.5)[:-1])
plt.plot(bins_mp, peace.quantile(.4)[:-1])
plt.plot(bins_mp, peace.quantile(.3)[:-1])
plt.plot(bins_mp, peace.quantile(.2)[:-1])
plt.plot(bins_mp, peace.quantile(.1)[:-1])
plt.scatter(las_data.distance_to_track, las_data.intensity, s=1)
plt.show()

curve_x = bins[:-1] + (bins[1:] - bins[:-1]) / 2
curve_y = peace.quantile(.95)[:-1]/1000

curve_x = las_data.distance_to_track
curve_y = las_data.intensity/1000


def func(x, a, b):
    return a * np.exp(-b * x)

def func(x, a, b):
    return a * x ** -b


popt, pcov = curve_fit(func, curve_x, curve_y, p0=[1., .5])

plt.scatter(curve_x, curve_y)
plt.plot(curve_x, func(curve_x, *popt), color='red')

# curve-normalized

norm_range = 80
las_data = las_data.assign(intensity_norm_exp=las_data.intensity * np.exp(0.00496 * (las_data.distance_to_track - norm_range)))
plt04 = las_data.plot.scatter(x='distance_to_track', y='intensity_norm_exp', s=1)

norm_range = 80
las_data = las_data.assign(intensity_norm_pow=las_data.intensity * las_data.distance_to_track ** 0.3085 * norm_range ** -0.3085)
plt05 = las_data.plot.scatter(x='distance_to_track', y='intensity_norm_exp', s=1)

# hmmm seems there are some issues still, but could be used as a starting point

# plot height vs. distance
plt01 = las_data.plot.scatter(x='distance_to_track', y='intensity', s=1)
plt01.set_ylim(0, 3000)
plt01.set_xlim(0, 120)
plt01.set(title="amplitude with distance", xlabel="distance to target (m)", ylabel="RiProcess amplitude")

plt02 = las_data.plot.scatter(x='distance_to_track', y='intensity_norm', s=1)
plt02.set_ylim(0, 3000)
plt02.set_xlim(0, 120)
plt02.set(title="amplitude*distance^2 with distance", xlabel="distance to target (m)",
          ylabel="RiProcess amplitude * (distance to target)^2/80^2")

plt03 = las_data.plot.scatter(x='distance_to_track', y='intensity_log')
plt03.set_ylim(6, 8)
plt03.set_xlim(0, 110)

# linear regression
# create regression object

# get x and y vectors

# calculate polynomial
z = np.polyfit(las_data.distance_to_track, las_data.intensity_log, 1)
f = np.poly1d(z)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(x, y, 'bo', label="Data")
plt.plot(x, f(x), 'b-', label="Polyfit")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
plt03 = plt.scatter(las_data.distance_to_track, las_data.intensity_log)
plt03 = plt.plot(las_data.distance_to_track, f(las_data.distance_to_track), color='red')
plt.show()

plt04 = plt.scatter(las_data.distance_to_track, las_data.intensity_log - f(las_data.distance_to_track))

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
