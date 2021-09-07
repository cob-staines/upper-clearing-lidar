import numpy as np
import pandas as pd
import h5py

### new run through: What I want is a ratio of canopy to ground points across scan anles

# file paths
las_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\LAS\\19_149_UF.las"
traj_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_all_traj.txt"
hdf5_path = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\LAS\\19_149_UF.hdf5"

# laslib.las_traj(las_in, traj_in, hdf5_path, chunksize=10000, keep_return='all', drop_class=None)

with h5py.File(hdf5_path, 'r') as hf:
    las_time = hf['lasData'][:, 0]
    traj_time = hf['trajData'][:, 0]

# first ground
# last ground
# first canopy


### old hat below

# # load las into hdf5
# laslib.las_to_hdf5(las_in, hdf5_path)

# read in las data
las_data = pd.read_hdf(hdf5_path, key='data', columns=['gps_time', 'num_returns', 'return_num', 'x', 'y', 'z', 'scan_angle_rank'])

# drop single returns
# multi_data = las_data[las_data.num_returns > 1]

q_fraction = 0.1
q_upper = las_data.gps_time.quantile(q=(0.7 + q_fraction/2))
q_lower = las_data.gps_time.quantile(q=(0.7 - q_fraction/2))
sub_data = las_data[(las_data.gps_time > q_lower) & (las_data.gps_time < q_upper)]

data = sub_data

# point-to-point

# group multiple_returns by gps_time
grouped = data.groupby('gps_time')

# count points for each gps_time
counts = grouped['return_num'].size()

#identify index of 1st and last return for each time
first_point_id = grouped.return_num.idxmin()
last_point_id = grouped.return_num.idxmax()

# concat to single data frame
time_data = pd.concat([counts, first_point_id, last_point_id], axis=1, ignore_index=False)
time_data.columns = ['point_count', 'first_point_id', 'last_point_id']

# filter to counts > 1 (even though all are multiple returns, some have only a single return in file due too clipping)
time_data = time_data[time_data.point_count > 1].reset_index()

# merge with first point id
time_data = time_data.merge(data[['x', 'y', 'z']], how='left',
                        left_on='first_point_id', right_index=True, validate='1:1')
time_data.columns = ['gps_time', 'point_count', 'first_point_id', 'last_point_id', 'x_1', 'y_1', 'z_1']

# merge with last point id
time_data = time_data.merge(data[['x', 'y', 'z']], how='left',
                        left_on='last_point_id', right_index=True, validate='1:1')
time_data.columns = ['gps_time', 'point_count', 'first_point_id', 'last_point_id', 'x_1', 'y_1', 'z_1', 'x_2', 'y_2', 'z_2']

# distance between points
p1 = np.array([time_data.x_1, time_data.y_1, time_data.z_1])
p2 = np.array([time_data.x_2, time_data.y_2, time_data.z_2])
squared_dist = np.sum((p1 - p2) ** 2, axis=0)
time_data = time_data.assign(first_last_dist_p2p=np.sqrt(squared_dist))

# angle from nadir
dp = p1 - p2
phi = np.arctan(np.sqrt(dp[0] ** 2 + dp[1] ** 2) / dp[2]) * 180 / np.pi  # in degrees
time_data = time_data.assign(angle_from_nadir_deg_p2p=phi)

# angle cw from north
theta = np.arctan2(dp[0], (dp[1])) * 180/np.pi
time_data = time_data.assign(angle_cw_from_north_deg_p2p=theta)

# merge back with las_data
#left = data[["gps_time", "scan_angle_rank"]].reset_index()
#p2p_data = left.merge(time_data[["gps_time", "first_last_dist", "angle_from_nadir_deg"]], on="gps_time", how="left", validate="m:1")
#p2p_data = p2p_data.set_index('index')
#p2p_data.columns = ["gps_time", "scan_angle_rank", "first_last_dist", "phi_p2p"]

left = data.reset_index()
p2p_data = left.merge(time_data, on="gps_time", how="left", validate="m:1")
p2p_data = p2p_data.set_index('index')

##############
# las_traj

las = data.assign(las=True)

# import trajectory
traj = pd.read_csv(traj_in)
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

# proper merge takes too long, instead keep track of index
outer = las[['gps_time', 'las']].append(traj, sort=False)
outer = outer.reset_index()
outer = outer.rename(columns={"index": "index_las"})

# order by gps time
outer = outer.sort_values(by="gps_time")
# set index as gps_time for nearest neighbor interpolation
outer = outer.set_index('gps_time')
# interpolate by nearest neighbor

interpolated = outer.interpolate(method="nearest")

# drop traj entries
interpolated = interpolated[interpolated['las']]
# reset to las index
interpolated = interpolated.set_index("index_las")
# drop las column
interpolated = interpolated[['easting_m', 'northing_m', 'height_m']]


# concatenate with index
merged = pd.concat([data, interpolated], axis=1, ignore_index=False)

# calculate point distance from track
p1 = np.array([merged.easting_m, merged.northing_m, merged.height_m])
p2 = np.array([merged.x, merged.y, merged.z])
squared_dist = np.sum((p1 - p2) ** 2, axis=0)
merged = merged.assign(distance_to_track=np.sqrt(squared_dist))

# calculate angle from nadir
dp = p1 - p2
phi = np.arctan(np.sqrt(dp[0] ** 2 + dp[1] ** 2) / dp[2]) * 180 / np.pi  # in degrees
merged = merged.assign(angle_from_nadir_deg=phi)

# angle cw from north
theta = np.arctan2(dp[0], (dp[1])) * 180/np.pi
merged = merged.assign(angle_cw_from_north_deg=theta)

las_traj = merged[["gps_time", "distance_to_track", "angle_from_nadir_deg"]]
las_traj.columns = ["gps_time_las_traj", "dist_las_traj", "phi_las_traj"]

######### concatinate all
merged.columns = ['gps_time_las_traj', 'num_returns', 'return_num', 'x_m', 'y_m', 'z_m',
       'scan_angle_rank_merged', 'easting_m', 'northing_m', 'height_m',
       'distance_to_track_traj', 'angle_from_nadir_deg_traj', 'angle_cw_from_north_deg_traj']

fin_data = pd.concat([p2p_data, merged], axis=1, ignore_index=False)
fin_data = fin_data[~np.isnan(fin_data.point_count)]

# plot
import matplotlib
matplotlib.use('TkAgg')  # use for interactive plotting
import matplotlib.pyplot as plt



plt.scatter(fin_data.angle_cw_from_north_deg_p2p, fin_data.angle_cw_from_north_deg_traj, s=1)
plt.scatter(fin_data.angle_from_nadir_deg_traj, fin_data.angle_from_nadir_deg_p2p, s=1, c=fin_data.gps_time)
plt.scatter(np.abs(fin_data.scan_angle_rank), fin_data.angle_from_nadir_deg_traj, s=1, c=fin_data.gps_time)
plt.scatter(np.abs(fin_data.scan_angle_rank), fin_data.angle_from_nadir_deg_p2p, s=1, c=fin_data.gps_time)

plt.scatter(fin_data.x, fin_data.x_1, s=1)
plt.scatter(fin_data.x, fin_data.x_2, s=1)
plt.scatter(fin_data.x, fin_data.x_m, s=1)
plt.scatter(fin_data.x, fin_data.easting_m, s=1)
plt.scatter((fin_data.x_1 - fin_data.x_2)/(fin_data.z_1 - fin_data.z_2), (fin_data.x - fin_data.easting_m)/(fin_data.z - fin_data.height_m), s=1)
plt.scatter((fin_data.y_1 - fin_data.y_2)/(fin_data.z_1 - fin_data.z_2), (fin_data.y - fin_data.northing_m)/(fin_data.z - fin_data.height_m), s=1)
plt.scatter(np.sqrt((fin_data.y_1 - fin_data.y_2) ** 2 + (fin_data.x_1 - fin_data.x_2) ** 2)/(fin_data.z_1 - fin_data.z_2), ((fin_data.y - fin_data.northing_m) ** 2 + (fin_data.x - fin_data.easting_m) ** 2)/(fin_data.z - fin_data.height_m), s=1)

plt.scatter(fin_data.easting_m, fin_data.height_m, c=fin_data.gps_time)

# filtering
cw_dif = (np.abs(fin_data.angle_cw_from_north_deg_p2p - fin_data.angle_cw_from_north_deg_traj) < 1)
plt.scatter(fin_data.angle_from_nadir_deg_traj[cw_dif], fin_data.angle_from_nadir_deg_p2p[cw_dif], s=1)

##############
# dask las_traj

import numpy as np
import pandas as pd

# file paths
las_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\LAS\\19_149_UF.las"
traj_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_all_traj.txt"
hdf5_path = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\LAS\\19_149_UF.hdf5"

# load las into hdf5
# laslib.las_to_hdf5(las_in, hdf5_path)
def las_traj(hdf5_path, traj_in):
    point_data = pd.read_hdf(hdf5_path, key='las_data', columns=['gps_time', 'x', 'y', 'z'])

    las_data = point_data.assign(las=True)

    # import trajectory
    traj = pd.read_csv(traj_in)
    # rename columns for consistency
    traj = traj.rename(columns={'Time[s]': "gps_time",
                                'Easting[m]': "easting_m",
                                'Northing[m]': "northing_m",
                                'Height[m]': "height_m"})
    # throw our pitch, roll, yaw
    traj = traj[['gps_time', 'easting_m', 'northing_m', 'height_m']]
    traj = traj.assign(las=False)

    # append traj to las, keeping track of las index
    outer = las_data[['gps_time', 'las']].append(traj, sort=False)
    outer = outer.reset_index()
    outer = outer.rename(columns={"index": "index_las"})

    # order by gps time
    outer = outer.sort_values(by="gps_time")

    # quality control

    # check first and last entries are traj
    if (outer.las.iloc[0] | outer.las.iloc[-1]):
        raise Exception('LAS data exists outside trajectory time frame -- Suspect LAS/trajectory file mismatch')

    # set index as gps_time
    outer = outer.set_index('gps_time')

    interpolated = outer.fillna(method='ffill')

    # drop traj entries
    interpolated = interpolated[interpolated['las']]
    # reset to las index
    interpolated = interpolated.set_index("index_las")
    # drop las column
    interpolated = interpolated[['easting_m', 'northing_m', 'height_m']]

    # concatenate with las_data horizontally by index
    merged = pd.concat([point_data, interpolated], axis=1, ignore_index=False)

    # calculate point distance from track
    p1 = np.array([merged.easting_m, merged.northing_m, merged.height_m])
    p2 = np.array([merged.x, merged.y, merged.z])
    squared_dist = np.sum((p1 - p2) ** 2, axis=0)
    merged = merged.assign(distance_from_sensor_m=np.sqrt(squared_dist))

    # calculate angle from nadir
    dp = p1 - p2
    phi = np.arctan(np.sqrt(dp[0] ** 2 + dp[1] ** 2) / dp[2]) * 180 / np.pi  # in degrees
    merged = merged.assign(angle_from_nadir_deg=phi)

    # angle cw from north
    theta = np.arctan2(dp[0], (dp[1])) * 180/np.pi
    merged = merged.assign(angle_cw_from_north_deg=theta)

    output = merged[["gps_time", "distance_from_sensor_m", "angle_from_nadir_deg", "angle_cw_from_north_deg"]]

    output.to_hdf(hdf5_path, key='las_traj', mode='r+', format='table')

# eval
las_data = pd.read_hdf(hdf5_path, key='las_data', columns=['gps_time', 'scan_angle_rank', 'intensity'])
las_traj = pd.read_hdf(hdf5_path, key='las_traj', columns=["distance_from_sensor_m", "angle_from_nadir_deg", "angle_cw_from_north_deg"])

merged = pd.concat([las_data, las_traj], axis=1, ignore_index=False)

# plot
import matplotlib
matplotlib.use('TkAgg')  # use for interactive plotting
import matplotlib.pyplot as plt

plt.scatter(np.abs(merged.scan_angle_rank), merged.angle_from_nadir_deg, s=1)
plt.scatter(merged.angle_from_nadir_deg, merged.intensity, s=1)
plt.scatter(merged.distance_from_sensor_m, merged.intensity, s=1)
