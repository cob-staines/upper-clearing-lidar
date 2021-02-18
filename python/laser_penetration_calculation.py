import laslib
import numpy as np
import pandas as pd
import h5py

# file paths
las_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\LAS\\19_149_UF.las"
traj_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_all_traj.txt"
hdf5_path = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\LAS\\19_149_UF.hdf5"

# laslib.las_traj(las_in, traj_in, hdf5_path, chunksize=10000, keep_return='all', drop_class=None)

with h5py.File(hdf5_path, 'r') as hf:
    las_data = hf['lasData'][:]
    traj_data = hf['trajData'][:]

las_pd = pd.DataFrame(data=las_data, index=None, columns=["gps_time", "x", "y", "z", "classification", "num_returns", "return_num"])
traj_pd = pd.DataFrame(data=traj_data, index=None, columns=["gps_time", "traj_x", "traj_y", "traj_z", "distance_from_sensor_m", "angle_from_nadir_deg", "angle_cw_from_north_deg"])

las_traj = las_pd + traj_pd.loc[:, 1:7]

# LAS classes:
# 1 -- unclassified
# 2 -- ground
# 5 -- vegetation
# 7 -- ground noise
# 8 -- vegetation noise
las_data = pd.read_hdf(hdf5_path, key='las_data', columns=['gps_time', 'classification', 'return_num', 'num_returns'])
las_traj = pd.read_hdf(hdf5_path, key='las_traj', columns=["distance_from_sensor_m", "angle_from_nadir_deg", "angle_cw_from_north_deg"])

las_data = pd.concat([las_data, las_traj], axis=1, ignore_index=False)

# drop noise
las_data = las_data[(las_data.classification != 7) & (las_data.classification != 8)]

# # filter by scanning pattern
# grid60 = (las_data.gps_time >= 324565) & (las_data.gps_time <= 324955)  # 20% of data
# grid120 = (las_data.gps_time >= 325588) & (las_data.gps_time <= 325800)  # 6% of data
# f_ladder = (las_data.gps_time >= 326992) & (las_data.gps_time <= 327260)  # 16% of data
# f_spin = (las_data.gps_time >= 325018) & (las_data.gps_time <= 325102)  # 18% of data
#
# las_data = las_data[grid120 | grid60 | f_ladder | f_spin]
#
# np.sum(f_spin)/las_data.__len__()

las_data = las_data.assign(afn_bin=las_data.angle_from_nadir_deg.astype(int))

# count by classification, first/last, and scan_angle
FG = las_data[(las_data.return_num == 1) & (las_data.classification == 2)].groupby('afn_bin').size()
FC = las_data[(las_data.return_num == 1) & (las_data.classification == 5)].groupby('afn_bin').size()
LG = las_data[(las_data.return_num == las_data.num_returns) & (las_data.classification == 2)].groupby('afn_bin').size()
LC = las_data[(las_data.return_num == las_data.num_returns) & (las_data.classification == 5)].groupby('afn_bin').size()
SG = las_data[(las_data.num_returns == 1) & (las_data.classification == 2)].groupby('afn_bin').size()
SC = las_data[(las_data.num_returns == 1) & (las_data.classification == 5)].groupby('afn_bin').size()
# concatinate
lpm = pd.concat([FG, FC, LG, LC, SG, SC], ignore_index=False, axis=1)
lpm.columns = ['FG', 'FC', 'LG', 'LC', 'SG', 'SC']

# want to know, for each sample, at each angle, what is the probability of returning a certain class
n_returns = las_data.groupby('afn_bin').size()
n_samples = las_data[(las_data.return_num == 1)].groupby('afn_bin').size()
n_samples_cum = np.nancumsum(n_samples)
nn = las_data.gps_time.nunique()

# plot counts
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


plt.plot(n_samples.index, n_samples)
plt.plot(n_samples.index, n_samples_cum/nn)

fig, ax = plt.subplots()
ax.plot(lpm.index, lpm.FG/n_samples, label="first ground")
ax.plot(lpm.index, lpm.FC/n_samples, label="first canopy")
ax.plot(lpm.index, lpm.LG/n_samples, label="last ground")
ax.plot(lpm.index, lpm.LC/n_samples, label="last canopy")
# plt.plot(lpm.index, lpm.SG/n_samples, label="single ground")  # same as first return ground FG
# plt.plot(lpm.index, lpm.SC/n_samples, label="single canopy")
ax.set(xlabel="scan angle (deg from nadir)", ylabel="relative frequency per sample", title="Frequency of returns by scan angle")
ax.grid()
ax.legend(title="return classification")
plt.show()

# calculate lpms -- 1 if all ground, 0 if all canopy
lpmf = lpm.FG/(lpm.FG + lpm.FC)
lpml = (lpm.FG + lpm.LG)/(lpm.FG + lpm.LG + lpm.FC)
lpmc = lpm.LG/(lpm.LG + lpm.FC)

# calculation of LMP_all
first_returns = las_data[las_data.return_num == 1]
first_returns = first_returns.assign(num_returns_inv=1/first_returns.num_returns)
first_returns = first_returns.groupby("afn_bin")
sum_inv_ret = first_returns.num_returns_inv.sum()

lpma = 1 - sum_inv_ret/n_samples

# path length correction... assume 1/cosine (beer-lambert)
plc = 1/np.cos(lpm.index*np.pi/180)

# plot lpms
plt.plot(lpmf.index, lpmf.values, label="LPM_firsts")
plt.plot(lpml.index, lpml.values, label="LPM_lasts")
plt.plot(lpmc.index, lpmc.values, label="LPM_canopy")
plt.plot(lpma.index, lpma.values, label="LPM_all")

# path-corrected lpms
fig, ax = plt.subplots()
ax.plot(lpmf.index, lpmf.values * plc, label="LPM_firsts")
ax.plot(lpml.index, lpml.values * plc, label="LPM_lasts")
ax.plot(lpmc.index, lpmc.values * plc, label="LPM_canopy")
# plt.plot(lpma.index, lpma.values * plc, label="LPM_all")  # not appropriate in this analysis
ax.set(xlabel="scan angle (deg from nadir)", ylabel="LPM value", title="Three laser penetration metics (LPMs) by scan angle")
ax.legend()
ax.grid()
plt.show()

# how do we account for clumping?
import pdal

las_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\LAS\\19_149_UF.las"

json = """
[
    "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/19_149/19_149_snow_off/OUTPUT_FILES/LAS/19_149_UF.las",
    {
        "type": "filters.sample",
        "radius": "0.15"
    },
    {
        "type": "writers.las",
        "filename": "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/19_149/19_149_snow_off/OUTPUT_FILES/LAS/19_149_UF_sampled.las"
    }
]
"""

pipeline = pdal.Pipeline(json)
count = pipeline.execute()
arrays = pipeline.arrays
metadata = pipeline.metadata
log = pipeline.log
