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
header = inFile.header
inFile.close()
las_data = las_data.assign(las=True)

# import trajectory
traj = pd.read_csv(filedir + traj_in)
# rename columns for consistency
traj = traj.rename(columns={'Time[s]': "gps_time",
                            'Easting[m]': "easting_m",
                            'Northing[m]': "northing_m",
                            'Height[m]': "height_m"})
# throw our pitch, roll, yaw (at least until needed later...)
traj = traj[['gps_time', 'easting_m', 'northing_m', 'height_m']]
traj = traj.assign(las=False)

# resample traj to las gps times and interpolate

# outer merge las and traj on gps_time
# outer = traj.merge(las_data.gps_time, on="gps_time", how="outer")
# more efficient to append and drop duplicates
outer = traj.append(las_data[['gps_time']], sort=False)
outer = outer.drop_duplicates(['gps_time'], keep='first')

# join too costly, instead keep track of index
outer = las_data[['gps_time', 'las']].append(traj, sort=False)
outer = outer.reset_index()
outer = outer.rename(columns={"index": "index_las"})

# order by gps time
outer = outer.sort_values(by="gps_time")
# set index as gps_time for nearest neighbor interpolation
outer = outer.set_index('gps_time')
# interpolate by nearest neighbor
interpolated = outer.interpolate(method="nearest")  # issues with other columns.... can we specify?
# resent index for clarity

interpolated = interpolated[interpolated['las']]
interpolated = interpolated.sort_values(by="index_las")
interpolated = interpolated.reset_index()
interpolated = interpolated[['easting_m', 'northing_m', 'height_m']]

merged = pd.concat([las_data, interpolated], axis=1)
merged = merged.drop('las', axis=1)

# calculate point distance from track
p1 = np.array([merged.easting_m, merged.northing_m, merged.height_m])
p2 = np.array([merged.x, merged.y, merged.z])
squared_dist = np.sum((p1 - p2) ** 2, axis=0)
merged = merged.assign(distance_to_track=np.sqrt(squared_dist))

# calculate angle from nadir
dp = p1 - p2
phi = np.arctan(np.sqrt(dp[0]**2 + dp[1]**2)/dp[2]) #*180/np.pi #for degrees
merged = merged.assign(angle_from_nadir=phi)

# --------------------------------SANDBOX---------------------------------
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


def func(x, a, b, c):
    return a * np.exp(-b * x) * x ** -c


popt, pcov = curve_fit(func, curve_x, curve_y, p0=[1000., .001, 1])

plt.scatter(las_data.distance_to_track, las_data.intensity/1000, s=1)
plt.scatter(curve_x, curve_y)
plt.plot(curve_x, func(curve_x, *popt), color='red')


def func(I, x1, x2, b, c):
    return I * np.exp(b * (x1 - x2)) * x1 ** c * x2 ** -c


b = 0.0012043585743089392
# b = popt[1]
c = 0.23473660663020066
# c = popt[2]
temp = func(las_data.intensity, las_data.distance_to_track, 80, b, c)
las_data = las_data.assign(intensity_range_norm=temp)

plt01 = las_data.plot.scatter(x='distance_to_track', y='intensity_range_norm', s=1)
plt01 = las_data.plot.scatter(x='angle_from_nadir', y='intensity_range_norm', s=1)

las_data = las_data.assign(intensity_angle_norm=las_data.intensity_range_norm / (np.cos(las_data.angle_from_nadir)) ** 2)

plt03 = las_data.plot.scatter(x='angle_from_nadir', y='intensity_angle_norm', s=1)

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
