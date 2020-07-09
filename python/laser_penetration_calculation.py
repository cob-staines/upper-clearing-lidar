las_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\LAS\\19_149_UF.las"
traj_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_all_traj.txt"
hdf5_path = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\LAS\\19_149_UF.hdf5"
hdf5_out = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\LAS\\19_149_UF.hdf5"

import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import numpy as np
import laslib
# plot classification counts by angle
import matplotlib
matplotlib.use('TkAgg')  # use for interactive plotting
import matplotlib.pyplot as plt

# initiate dask client
cluster = LocalCluster()
client = Client(cluster)

# convert las to hdf5
laslib.las_to_hdf5(las_in, hdf5_path)
# read hdf5_path into dask
#las_data = dd.read_hdf(hdf5_path, key='data', columns=['gps_time', 'num_returns', 'return_num'])
las_data = pd.read_hdf(hdf5_path, key='data', columns=['gps_time', 'num_returns', 'return_num', 'x', 'y', 'z', 'scan_angle_rank'])

# take subset of data for speed
q_fraction = 0.1
q_upper = las_data.gps_time.quantile(q=(0.5 + q_fraction/2))
q_lower = las_data.gps_time.quantile(q=(0.5 - q_fraction/2))
las_data = las_data[(las_data.gps_time > q_lower) & (las_data.gps_time < q_upper)]

# drop single returns
multiple_returns = las_data[['gps_time', 'return_num', 'scan_angle_rank']][las_data.num_returns > 1]

# group multiple_returns by gps_time
grouped = multiple_returns.groupby('gps_time')
# count points for each gps_time
counts = grouped['return_num'].size()
#identify index of 1st and last return for each time
first_point_id = grouped.return_num.idxmin()
last_point_id = grouped.return_num.idxmax()

# concat to single data frame
time_data = pd.concat([counts, first_point_id, last_point_id], axis=1, ignore_index=False)
time_data.columns = ['point_count', 'first_point_id', 'last_point_id']
# filter to counts > 1
time_data = time_data[time_data.point_count > 1].reset_index()

time_data = time_data.merge(las_data[['x', 'y', 'z']], how='left',
                        left_on='first_point_id', right_index=True, validate='1:1')
time_data.columns = ['gps_time', 'point_count', 'first_point_id', 'last_point_id', 'x_1', 'y_1', 'z_1']
time_data = time_data.merge(las_data[['x', 'y', 'z']], how='left',
                        left_on='last_point_id', right_index=True, validate='1:1')
time_data.columns = ['gps_time', 'point_count', 'first_point_id', 'last_point_id', 'x_1', 'y_1', 'z_1', 'x_2', 'y_2', 'z_2']
# calculate phi and theta
# calculate point distance from track
p1 = np.array([time_data.x_1, time_data.y_1, time_data.z_1])
p2 = np.array([time_data.x_2, time_data.y_2, time_data.z_2])
squared_dist = np.sum((p1 - p2) ** 2, axis=0)
time_data = time_data.assign(first_last_dist=np.sqrt(squared_dist))

# calculate angle from nadir
dp = p1 - p2
phi = np.arctan(np.sqrt(dp[0] ** 2 + dp[1] ** 2) / dp[2]) * 180 / np.pi  # in degrees
time_data = time_data.assign(angle_from_nadir_deg=phi)
theta = np.arctan(dp[0]/(dp[1] + 0.00001)) * 180/np.pi
time_data = time_data.assign(angle_cw_from_north_deg=theta)

# merge with multiple_returns
mr2 = multiple_returns.reset_index()
angle_comp = mr2[['index', 'gps_time', 'scan_angle_rank']].merge(time_data[['gps_time', 'angle_from_nadir_deg']], on="gps_time", how="inner", validate="m:1")
angle_comp = angle_comp.set_index('index')
angle_comp.columns = ["gps_time", "scan_angle_rank", "phi_fl"]

# las_traj
peace = laslib.las_traj(las_in, traj_in)

df = pd.merge(angle_comp, peace[['gps_time', 'angle_from_nadir_deg']], how='left', left_index=True, right_index=True, validate='1:1')

# plotting

plt.scatter(df.phi_fl, df.angle_from_nadir_deg, s=1)
plt.xlabel('point-to-point angle from nadir (deg)')
plt.ylabel('platform-to-point angle from nadir (deg)')

plt.scatter(df.scan_angle_rank, df.phi_fl, s=1)
plt.xlabel('scan angel rank (deg)')
plt.ylabel('point-to-point angle from nadir (deg)')

plt.scatter(df.scan_angle_rank, df.angle_from_nadir_deg, s=1)
plt.xlabel('scan angel rank (deg)')
plt.ylabel('platform-to-point angle from nadir (deg)')

# LPM scan angle analysis
# 1 -- unclassified
# 2 -- ground
# 5 -- vegetation
# 7 -- ground noise
# 8 -- vegetation noise
las_data = pd.read_hdf(hdf5_path, key='data', columns=['gps_time', 'classification', 'return_num', 'num_returns', 'scan_angle_rank'])
# drop noise
las_data = las_data[(las_data.classification != 7) & (las_data.classification != 8)]
# las_data.groupby('classification').size()
# take abs of scan angle rank
las_data = las_data.assign(scan_angle=np.abs(las_data.scan_angle_rank))

# filter by scanning pattern
grid60 = (las_data.gps_time >= 324565) & (las_data.gps_time <= 324955)  # 20% of data
grid120 = (las_data.gps_time >= 325588) & (las_data.gps_time <= 325800)  # 6% of data
f_ladder = (las_data.gps_time >= 326992) & (las_data.gps_time <= 327260)  # 16% of data
f_spin = (las_data.gps_time >= 325018) & (las_data.gps_time <= 325102)  # 18% of data

las_data = las_data[grid120 | grid60 | f_ladder | f_spin]

np.sum(f_spin)/f_spin.__len__()

# count by classification, first/last, and scan_angle
FG = las_data[(las_data.return_num == 1) & (las_data.classification == 2)].groupby('scan_angle').size()
FC = las_data[(las_data.return_num == 1) & (las_data.classification == 5)].groupby('scan_angle').size()
LG = las_data[(las_data.return_num == las_data.num_returns) & (las_data.classification == 2)].groupby('scan_angle').size()
LC = las_data[(las_data.return_num == las_data.num_returns) & (las_data.classification == 5)].groupby('scan_angle').size()
SG = las_data[(las_data.num_returns == 1) & (las_data.classification == 2)].groupby('scan_angle').size()
SC = las_data[(las_data.num_returns == 1) & (las_data.classification == 5)].groupby('scan_angle').size()
# concatinate
lpm = pd.concat([FG, FC, LG, LC, SG, SC], ignore_index=False, axis=1)
lpm.columns = ['FG', 'FC', 'LG', 'LC', 'SG', 'SC']

# want to know, for each sample, at each angle, what is the probability of returning a certain class
n_returns = las_data.groupby('scan_angle').size()
n_samples = las_data[(las_data.return_num == 1)].groupby('scan_angle').size()
n_samples_cum = np.nancumsum(n_samples)
nn = las_data.gps_time.nunique()

# plot counts
plt.plot(n_samples.index, n_samples)
plt.plot(n_samples.index, n_samples_cum/nn)
plt.plot(lpm.index, lpm.FG/n_samples, label="first ground")
plt.plot(lpm.index, lpm.FC/n_samples, label="first canopy")
plt.plot(lpm.index, lpm.LG/n_samples, label="last ground")
plt.plot(lpm.index, lpm.LC/n_samples, label="last canopy")
# plt.plot(lpm.index, lpm.SG/n_samples, label="single ground")  # same as first return ground FG
# plt.plot(lpm.index, lpm.SC/n_samples, label="single canopy")
plt.legend(title="return classification")
plt.title("Relative probability of returns by scan angle")
plt.show()

# calculate lpms -- 1 if all ground, 0 if all canopy
lpmf = lpm.FG/(lpm.FG + lpm.FC)
lpml = (lpm.FG + lpm.LG)/(lpm.FG + lpm.LG + lpm.FC)
lpmc = lpm.LG/(lpm.LG + lpm.FC)

# calculation of LMP_all
first_returns = las_data[las_data.return_num == 1]
first_returns = first_returns.assign(num_returns_inv=1/first_returns.num_returns)
first_returns = first_returns.groupby("scan_angle")
sum_inv_ret = first_returns.num_returns_inv.sum()

lpma = 1 - sum_inv_ret/n_samples

# path length correction... assume 1/cosine (beer-lambert)
plc = 1/np.cos(lpm.index*np.pi/180)

# plot lpms
plt.plot(lpmf.index, lpmf.values, label="LPM_firsts")
plt.plot(lpml.index, lpml.values, label="LPM_lasts")
plt.plot(lpmc.index, lpmc.values, label="LPM_canopy")
plt.plot(lpma.index, lpma.values, label="LPM_all")
plt.plot(lpmf.index, lpmf.values * plc, label="LPM_firsts")
plt.plot(lpml.index, lpml.values * plc, label="LPM_lasts")
plt.plot(lpmc.index, lpmc.values * plc, label="LPM_canopy")
plt.plot(lpma.index, lpma.values * plc, label="LPM_all")  # not appropriate in this case?
plt.legend()
plt.show()


# how do we account for clumping?