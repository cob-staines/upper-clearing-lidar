import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
import tifffile as tif
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

# file paths
las_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\LAS\\19_149_UF.las"
traj_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_all_traj.txt"
hdf5_path = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\LAS\\19_149_UF.hdf5"

plot_out_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\graphics\\thesis_graphics\\transmittance\\"


# laslib.las_traj(las_in, traj_in, hdf5_path, chunksize=10000, keep_return='all', drop_class=None)

with h5py.File(hdf5_path, 'r') as hf:
    las_data = hf['lasData'][:]
    traj_data = hf['trajData'][:]

las_pd = pd.DataFrame(data=las_data, index=None, columns=["gps_time", "x", "y", "z", "classification", "num_returns", "return_num"])
traj_pd = pd.DataFrame(data=traj_data, index=None, columns=["gps_time", "traj_x", "traj_y", "traj_z", "distance_from_sensor_m", "angle_from_nadir_deg", "angle_cw_from_north_deg"])

# indexing
a_bins = np.linspace(0, 90, 91) - 0.5

# first return ground
fg_index = (las_pd.classification == 2.) & (las_pd.return_num == 1.0)
fg_phi = traj_pd.angle_from_nadir_deg[fg_index].values
fg_bins = np.digitize(fg_phi, bins=a_bins)
fg_ang, fg_count = np.unique(fg_bins, return_counts=True)
fg = pd.DataFrame({"phi": fg_ang - 1,
                   "fg_count": fg_count})

# last return ground
lg_index = (las_pd.classification == 2.) & (las_pd.return_num == las_pd.num_returns)
lg_phi = traj_pd.angle_from_nadir_deg[lg_index].values
lg_bins = np.digitize(lg_phi, bins=a_bins)
lg_ang, lg_count = np.unique(lg_bins, return_counts=True)
lg = pd.DataFrame({"phi": lg_ang - 1,
                   "lg_count": lg_count})

# first return canopy
fc_index = (las_pd.classification == 5.) & (las_pd.return_num == 1.0)
fc_phi = traj_pd.angle_from_nadir_deg[fc_index].values
fc_bins = np.digitize(fc_phi, bins=a_bins)
fc_ang, fc_count = np.unique(fc_bins, return_counts=True)
fc = pd.DataFrame({"phi": fc_ang - 1,
                   "fc_count": fc_count})

lpm = pd.DataFrame({"phi": a_bins[1:] - 0.5})
lpm = pd.merge(lpm, fg, on="phi", how="left")
lpm = pd.merge(lpm, lg, on="phi", how="left")
lpm = pd.merge(lpm, fc, on="phi", how="left")

# calculate lpms by angle bin
lpm.loc[:, "lpmf"] = (lpm.fg_count) / (lpm.fg_count + lpm.fc_count)
lpm.loc[:, "lpml"] = (lpm.fg_count + lpm.lg_count) / (lpm.fg_count + lpm.lg_count + lpm.fc_count)
lpm.loc[:, "lpmc"] = (lpm.lg_count) / (lpm.lg_count + lpm.fc_count)

# estimate G
lpm.loc[:, "g_lpmf"] = -np.log(lpm.lpmf) * np.cos((lpm.phi) * np.pi / 180)
lpm.loc[:, "g_lpml"] = -np.log(lpm.lpml) * np.cos((lpm.phi) * np.pi / 180)
lpm.loc[:, "g_lpmc"] = -np.log(lpm.lpmc) * np.cos((lpm.phi) * np.pi / 180)

# plot
plt.plot(lpm.phi, lpm.lpmf, label="lpmf")
plt.plot(lpm.phi, lpm.lpml, label="lpml")
plt.plot(lpm.phi, lpm.lpmc, label="lpmc")

plt.plot(lpm.phi, lpm.g_lpmf, label="g_lpmf")
plt.plot(lpm.phi, lpm.g_lpml, label="g_lpml")
plt.plot(lpm.phi, lpm.g_lpmc, label="g_lpmc")

plt.legend()

#####
# # compare with same from ray sampling
# batch_dir = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_uf_r.25_px181_snow_off_dem_offset.25\\outputs\\'
# scaling_coef = 0.38686933  # snow_off
# canopy = "snow_off"

batch_dir = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_uf_r.25_px181_snow_on_dem_offset.25\\outputs\\'
scaling_coef = 0.37181197  # snow_on
canopy = "snow_on"

# load canopy stats
df_25_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\merged_data_products\\merged_uf_r.25m_canopy_19_149_median-snow.csv'
df_25 = pd.read_csv(df_25_in)

stat_file = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\transmittance_stats\\'

# load img meta
hemimeta = pd.read_csv(batch_dir + 'rshmetalog.csv')
imsize = hemimeta.img_size_px[0]
# load image pixel angle lookup
angle_lookup = pd.read_csv(batch_dir + "phi_theta_lookup.csv")
# build phi image (in radians)
phi = np.full((imsize, imsize), np.nan)
phi[(np.array(angle_lookup.y_index), np.array(angle_lookup.x_index))] = angle_lookup.phi
# build theta image (in radians)
theta = np.full((imsize, imsize), np.nan)
theta[(np.array(angle_lookup.y_index), np.array(angle_lookup.x_index))] = angle_lookup.theta

imrange = (phi <= np.pi / 2)

param_thresh = 1
set_param = np.random.random(len(hemimeta))
hemimeta.loc[:, 'training_set'] = set_param < param_thresh
# build hemiList from training_set only
hemiList = hemimeta.loc[hemimeta.training_set, :].reset_index()

# load contact number
imstack = np.full([imsize, imsize, len(hemiList)], np.nan)
for ii in tqdm(range(0, len(hemiList))):
    imstack[:, :, ii] = tif.imread(batch_dir + hemiList.file_name[ii])[:, :, 0] * scaling_coef

intnum = 1

# rerun corcoef_e with optimized transmission scalar
hemi_mean_tx = np.full((imsize, imsize), np.nan)
hemi_sd_tx = np.full((imsize, imsize), np.nan)
hemi_mean_cn = np.full((imsize, imsize), np.nan)
hemi_sd_cn = np.full((imsize, imsize), np.nan)
for ii in range(0, imsize):
    for jj in range(0, imsize):
        if imrange[jj, ii]:
            cn = intnum * imstack[jj, ii, :]
            tx = np.exp(-cn)
            hemi_mean_tx[jj, ii] = np.mean(tx)
            hemi_sd_tx[jj, ii] = np.std(tx)
            hemi_mean_cn[jj, ii] = np.mean(cn)
            hemi_sd_cn[jj, ii] = np.std(cn)

    print(ii)

def add_polar_axes(r_label_color="black"):
    # create polar axes and labels
    ax = fig.add_subplot(111, polar=True, label="polar")
    ax.set_facecolor("None")
    ax.set_rmax(90)
    ax.set_rgrids(np.linspace(0, 90, 7), labels=['', '15$^\circ$', '30$^\circ$', '45$^\circ$', '60$^\circ$', '75$^\circ$', '90$^\circ$'], angle=315)
    [t.set_color(r_label_color) for t in ax.yaxis.get_ticklabels()]
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.linspace(0, 360, 4, endpoint=False), labels=['N\n  0$^\circ$', 'W\n  270$^\circ$', 'S\n  180$^\circ$', 'E\n  90$^\circ$'])


## plot mean tx
fig = plt.figure(figsize=(7, 7))
#create axes in the background to show cartesian image
ax0 = fig.add_subplot(111)
im = ax0.imshow(hemi_mean_tx, cmap='Greys_r', clim=(0, 1))
ax0.axis("off")

add_polar_axes(r_label_color="white")

# add colorbar
fig.subplots_adjust(top=0.95, left=0.1, right=0.75, bottom=0.05)
cbar_ax = fig.add_axes([0.85, 0.20, 0.03, 0.6])
fig.colorbar(im, cax=cbar_ax)
fig.savefig(plot_out_dir + "uf_tx_mean_" + canopy + ".png")

## plot sd tx
fig = plt.figure(figsize=(7, 7))
#create axes in the background to show cartesian image
ax0 = fig.add_subplot(111)
im = ax0.imshow(hemi_sd_tx, cmap='Greys_r')
ax0.axis("off")

add_polar_axes()

# add colorbar
fig.subplots_adjust(top=0.95, left=0.1, right=0.75, bottom=0.05)
cbar_ax = fig.add_axes([0.85, 0.20, 0.03, 0.6])
fig.colorbar(im, cax=cbar_ax)
fig.savefig(plot_out_dir + "uf_tx_sd_" + canopy + ".png")

# summarize by angle band
phi_step = 1  # in degrees
phi_bin = np.floor(phi * 180 / (np.pi * phi_step)) * phi_step  # similar to ceil, except for int values
bins = np.unique(phi_bin)
bins = bins[~np.isnan(bins)]
bin_count = len(bins)

theta_step = 90 # in degrees
theta_bin = (theta * 180 / np.pi + 45)
theta_bin[theta_bin >= 360] -= 360
theta_bin = np.floor(theta_bin / theta_step) * theta_step
t_bins = np.unique(theta_bin)  # N, E, S, W
t_bins = t_bins[~np.isnan(t_bins)]
t_bin_count = len(t_bins)

# all theta
tx_bin_mean = np.full(bin_count, np.nan)
tx_bin_sd = np.full(bin_count, np.nan)

cn_bin_mean = np.full(bin_count, np.nan)
cn_bin_sd = np.full(bin_count, np.nan)
for ii in range(0, bin_count):
    bb = bins[ii]
    mask = (phi_bin == bb)
    tx_bin_mean[ii] = np.mean(hemi_mean_tx[mask])
    tx_bin_sd[ii] = np.mean(hemi_sd_tx[mask])

    cn_bin_mean[ii] = np.mean(hemi_mean_cn[mask])
    cn_bin_sd[ii] = np.mean(hemi_sd_cn[mask])
    # tx_std_bin_means[:, ii] = np.mean(tx_std_stack_long[:, mask], axis=1)

fig = plt.figure()
fig.subplots_adjust(top=0.90, bottom=0.15, left=0.15)
ax1 = fig.add_subplot(111)
plt.plot(bins, tx_bin_mean)
lower_bound = tx_bin_mean - tx_bin_sd
upper_bound = tx_bin_mean + tx_bin_sd
plt.fill_between(bins, lower_bound, upper_bound, alpha=0.5)
plt.xlim(0, 90)
plt.ylim(0, 1)
ax1.set_ylabel("Light transmittance [-]")
ax1.set_xlabel("Angle from zenith [$^{\circ}$]")
fig.savefig(plot_out_dir + "uf_tx_zenith_all_" + canopy + ".png")

# by direction
tx_dir_bin_mean = np.full([bin_count, t_bin_count], np.nan)
tx_dir_bin_sd = np.full([bin_count, t_bin_count], np.nan)

cn_dir_bin_mean = np.full([bin_count, t_bin_count], np.nan)
cn_dir_bin_sd = np.full([bin_count, t_bin_count], np.nan)
for jj in range(0, t_bin_count):
    tt = t_bins[jj]
    for ii in range(0, bin_count):
        bb = bins[ii]
        mask = (phi_bin == bb) & (theta_bin == tt)
        tx_dir_bin_mean[ii, jj] = np.mean(hemi_mean_tx[mask])
        tx_dir_bin_sd[ii, jj] = np.mean(hemi_sd_tx[mask])

        cn_dir_bin_mean[ii, jj] = np.mean(hemi_mean_cn[mask])
        cn_dir_bin_sd[ii, jj] = np.mean(hemi_sd_cn[mask])


fig = plt.figure()
fig.subplots_adjust(top=0.90, bottom=0.15, left=0.15)
ax1 = fig.add_subplot(111)
plt.plot(bins, tx_dir_bin_mean[:, 0], label="North")
plt.plot(bins, tx_dir_bin_mean[:, 1], label="East")
plt.plot(bins, tx_dir_bin_mean[:, 2], label="South")
plt.plot(bins, tx_dir_bin_mean[:, 3], label="West")
plt.legend(loc="upper right")

lower_bound = tx_dir_bin_mean - tx_dir_bin_sd
upper_bound = tx_dir_bin_mean + tx_dir_bin_sd

plt.fill_between(bins, lower_bound[:, 0], upper_bound[:, 0], alpha=0.25)
plt.fill_between(bins, lower_bound[:, 1], upper_bound[:, 1], alpha=0.25)
plt.fill_between(bins, lower_bound[:, 2], upper_bound[:, 2], alpha=0.25)
plt.fill_between(bins, lower_bound[:, 3], upper_bound[:, 3], alpha=0.25)

plt.xlim(0, 90)
plt.ylim(0, 1)
ax1.set_ylabel("Light transmittance [-]")
ax1.set_xlabel("Angle from zenith [$^{\circ}$]")
fig.savefig(plot_out_dir + "uf_tx_zenith_nesw_" + canopy + ".png")


# combine into df
if canopy == "snow_off":
    tx_all = pd.DataFrame({"phi": bins})
    tx_all.loc[:, "tx_mean_all_snow_off"] = tx_bin_mean
    tx_all.loc[:, "tx_mean_n_snow_off"] = tx_dir_bin_mean[:, 0]
    tx_all.loc[:, "tx_mean_e_snow_off"] = tx_dir_bin_mean[:, 1]
    tx_all.loc[:, "tx_mean_s_snow_off"] = tx_dir_bin_mean[:, 2]
    tx_all.loc[:, "tx_mean_w_snow_off"] = tx_dir_bin_mean[:, 3]
    tx_all.loc[:, "tx_sd_all_snow_off"] = tx_bin_sd
    tx_all.loc[:, "tx_sd_n_snow_off"] = tx_dir_bin_sd[:, 0]
    tx_all.loc[:, "tx_sd_e_snow_off"] = tx_dir_bin_sd[:, 1]
    tx_all.loc[:, "tx_sd_s_snow_off"] = tx_dir_bin_sd[:, 2]
    tx_all.loc[:, "tx_sd_w_snow_off"] = tx_dir_bin_sd[:, 3]
    tx_all.loc[:, "cn_mean_all_snow_off"] = cn_bin_mean
    tx_all.loc[:, "cn_mean_n_snow_off"] = cn_dir_bin_mean[:, 0]
    tx_all.loc[:, "cn_mean_e_snow_off"] = cn_dir_bin_mean[:, 1]
    tx_all.loc[:, "cn_mean_s_snow_off"] = cn_dir_bin_mean[:, 2]
    tx_all.loc[:, "cn_mean_w_snow_off"] = cn_dir_bin_mean[:, 3]
    tx_all.loc[:, "cn_sd_all_snow_off"] = cn_bin_sd
    tx_all.loc[:, "cn_sd_n_snow_off"] = cn_dir_bin_sd[:, 0]
    tx_all.loc[:, "cn_sd_e_snow_off"] = cn_dir_bin_sd[:, 1]
    tx_all.loc[:, "cn_sd_s_snow_off"] = cn_dir_bin_sd[:, 2]
    tx_all.loc[:, "cn_sd_w_snow_off"] = cn_dir_bin_sd[:, 3]
    # tx_all.loc[:, "g_all_snow_off"] = tx_all.cn_mean_all_snow_off * np.cos((tx_all.phi) * np.pi / 180) / np.mean(df_25.lrs_lai_2000)
    # tx_all.loc[:, "g_n_snow_off"] = tx_all.cn_mean_n_snow_off * np.cos((tx_all.phi) * np.pi / 180) / np.mean(df_25.lrs_lai_2000)
    # tx_all.loc[:, "g_e_snow_off"] = tx_all.cn_mean_e_snow_off * np.cos((tx_all.phi) * np.pi / 180) / np.mean(df_25.lrs_lai_2000)
    # tx_all.loc[:, "g_s_snow_off"] = tx_all.cn_mean_s_snow_off * np.cos((tx_all.phi) * np.pi / 180) / np.mean(df_25.lrs_lai_2000)
    # tx_all.loc[:, "g_w_snow_off"] = tx_all.cn_mean_w_snow_off * np.cos((tx_all.phi) * np.pi / 180) / np.mean(df_25.lrs_lai_2000)
    tx_all.loc[:, "g_all_snow_off"] = -np.log(tx_all.tx_mean_all_snow_off) * np.cos((tx_all.phi) * np.pi / 180) / np.mean(df_25.lrs_lai_2000)
    tx_all.loc[:, "g_n_snow_off"] = -np.log(tx_all.tx_mean_n_snow_off) * np.cos((tx_all.phi) * np.pi / 180) / np.mean(df_25.lrs_lai_2000)
    tx_all.loc[:, "g_e_snow_off"] = -np.log(tx_all.tx_mean_e_snow_off) * np.cos((tx_all.phi) * np.pi / 180) / np.mean(df_25.lrs_lai_2000)
    tx_all.loc[:, "g_s_snow_off"] = -np.log(tx_all.tx_mean_s_snow_off) * np.cos((tx_all.phi) * np.pi / 180) / np.mean(df_25.lrs_lai_2000)
    tx_all.loc[:, "g_w_snow_off"] = -np.log(tx_all.tx_mean_w_snow_off) * np.cos((tx_all.phi) * np.pi / 180) / np.mean(df_25.lrs_lai_2000)

    tx_all.to_csv(stat_file + 'transmittance_stats.csv', index=False)
elif canopy == "snow_on":
    tx_all = pd.read_csv(stat_file + 'transmittance_stats.csv')

    tx_all.loc[:, "tx_mean_all_snow_on"] = tx_bin_mean
    tx_all.loc[:, "tx_mean_n_snow_on"] = tx_dir_bin_mean[:, 0]
    tx_all.loc[:, "tx_mean_e_snow_on"] = tx_dir_bin_mean[:, 1]
    tx_all.loc[:, "tx_mean_s_snow_on"] = tx_dir_bin_mean[:, 2]
    tx_all.loc[:, "tx_mean_w_snow_on"] = tx_dir_bin_mean[:, 3]
    tx_all.loc[:, "tx_sd_all_snow_on"] = tx_bin_sd
    tx_all.loc[:, "tx_sd_n_snow_on"] = tx_dir_bin_sd[:, 0]
    tx_all.loc[:, "tx_sd_e_snow_on"] = tx_dir_bin_sd[:, 1]
    tx_all.loc[:, "tx_sd_s_snow_on"] = tx_dir_bin_sd[:, 2]
    tx_all.loc[:, "tx_sd_w_snow_on"] = tx_dir_bin_sd[:, 3]
    tx_all.loc[:, "cn_mean_all_snow_on"] = cn_bin_mean
    tx_all.loc[:, "cn_mean_n_snow_on"] = cn_dir_bin_mean[:, 0]
    tx_all.loc[:, "cn_mean_e_snow_on"] = cn_dir_bin_mean[:, 1]
    tx_all.loc[:, "cn_mean_s_snow_on"] = cn_dir_bin_mean[:, 2]
    tx_all.loc[:, "cn_mean_w_snow_on"] = cn_dir_bin_mean[:, 3]
    tx_all.loc[:, "cn_sd_all_snow_on"] = cn_bin_sd
    tx_all.loc[:, "cn_sd_n_snow_on"] = cn_dir_bin_sd[:, 0]
    tx_all.loc[:, "cn_sd_e_snow_on"] = cn_dir_bin_sd[:, 1]
    tx_all.loc[:, "cn_sd_s_snow_on"] = cn_dir_bin_sd[:, 2]
    tx_all.loc[:, "cn_sd_w_snow_on"] = cn_dir_bin_sd[:, 3]
    # tx_all.loc[:, "g_all_snow_on"] = tx_all.cn_mean_all_snow_on * np.cos((tx_all.phi) * np.pi / 180) / np.mean(df_25.lrs_lai_2000_snow_on)
    # tx_all.loc[:, "g_n_snow_on"] = tx_all.cn_mean_n_snow_on * np.cos((tx_all.phi) * np.pi / 180) / np.mean(df_25.lrs_lai_2000_snow_on)
    # tx_all.loc[:, "g_e_snow_on"] = tx_all.cn_mean_e_snow_on * np.cos((tx_all.phi) * np.pi / 180) / np.mean(df_25.lrs_lai_2000_snow_on)
    # tx_all.loc[:, "g_s_snow_on"] = tx_all.cn_mean_s_snow_on * np.cos((tx_all.phi) * np.pi / 180) / np.mean(df_25.lrs_lai_2000_snow_on)
    # tx_all.loc[:, "g_w_snow_on"] = tx_all.cn_mean_w_snow_on * np.cos((tx_all.phi) * np.pi / 180) / np.mean(df_25.lrs_lai_2000_snow_on)
    tx_all.loc[:, "g_all_snow_on"] = -np.log(tx_all.tx_mean_all_snow_on) * np.cos((tx_all.phi) * np.pi / 180) / np.mean(df_25.lrs_lai_2000_snow_on)
    tx_all.loc[:, "g_n_snow_on"] = -np.log(tx_all.tx_mean_n_snow_on) * np.cos((tx_all.phi) * np.pi / 180) / np.mean(df_25.lrs_lai_2000_snow_on)
    tx_all.loc[:, "g_e_snow_on"] = -np.log(tx_all.tx_mean_e_snow_on) * np.cos((tx_all.phi) * np.pi / 180) / np.mean(df_25.lrs_lai_2000_snow_on)
    tx_all.loc[:, "g_s_snow_on"] = -np.log(tx_all.tx_mean_s_snow_on) * np.cos((tx_all.phi) * np.pi / 180) / np.mean(df_25.lrs_lai_2000_snow_on)
    tx_all.loc[:, "g_w_snow_on"] = -np.log(tx_all.tx_mean_w_snow_on) * np.cos((tx_all.phi) * np.pi / 180) / np.mean(df_25.lrs_lai_2000_snow_on)

    tx_all.to_csv(stat_file + 'transmittance_stats.csv', index=False)

tx_all = pd.read_csv(stat_file + 'transmittance_stats.csv')

fig = plt.figure()
fig.subplots_adjust(top=0.90, bottom=0.15, left=0.15)
ax1 = fig.add_subplot(111)
plt.plot(90 - tx_all.phi, tx_all.g_all_snow_off, label="Snow off")
# plt.plot(90 - tx_all.phi, tx_all.g_n_snow_off, label="Snow off north")
# plt.plot(90 - tx_all.phi, tx_all.g_e_snow_off, label="Snow off east")
# plt.plot(90 - tx_all.phi, tx_all.g_s_snow_off, label="Snow off south")
# plt.plot(90 - tx_all.phi, tx_all.g_w_snow_off, label="Snow off west")
plt.plot(90 - tx_all.phi, tx_all.g_all_snow_on, label="Snow on")
# plt.plot(90 - tx_all.phi, tx_all.g_n_snow_on, label="Snow on north")
# plt.plot(90 - tx_all.phi, tx_all.g_e_snow_on, label="Snow on east")
# plt.plot(90 - tx_all.phi, tx_all.g_s_snow_on, label="Snow on south")
# plt.plot(90 - tx_all.phi, tx_all.g_w_snow_on, label="Snow on west")
plt.xlim([0, 40])
plt.ylim([0, 0.7])
plt.legend()
ax1.set_ylabel("Extinction efficiency [-]")
ax1.set_xlabel("Elevation angle [$^{\circ}$]")
fig.savefig(plot_out_dir + "uf_g_elevation.png")

bin_mid = (np.array([1, 2, 3, 4]) - .5) * 15

g_hemi_1 = -np.log(np.mean(df_25.transmission_s_1)) * np.cos((bin_mid[0]) * np.pi / 180) / np.mean(df_25.lai_s_cc)
g_hemi_2 = -np.log(np.mean(df_25.transmission_s_2)) * np.cos((bin_mid[1]) * np.pi / 180) / np.mean(df_25.lai_s_cc)
g_hemi_3 = -np.log(np.mean(df_25.transmission_s_3)) * np.cos((bin_mid[2]) * np.pi / 180) / np.mean(df_25.lai_s_cc)
g_hemi_4 = -np.log(np.mean(df_25.transmission_s_4)) * np.cos((bin_mid[3]) * np.pi / 180) / np.mean(df_25.lai_s_cc)
# g_hemi_5 = -np.log(np.mean(df_25.transmission_s_5)) * np.cos((bin_mid[4]) * np.pi / 180) / np.mean(df_25.lai_s_cc)
g_hemi = np.array([g_hemi_1, g_hemi_2, g_hemi_3, g_hemi_4])

def gx(xx, phi):
    k = np.sqrt(xx ** 2 + np.tan(phi * np.pi / 180) ** 2) / (xx + 1.702 * (xx + 1.12) ** -0.708)
    return k * np.cos(phi * np.pi / 180)

gx_0 = gx(0, tx_all.phi)
gx_h5 = gx(0.5, tx_all.phi)
gx_h75 = gx(0.75, tx_all.phi)
gx_1 = gx(1, tx_all.phi)
gx_inf = gx(2 ** 31, tx_all.phi)

fig = plt.figure()
fig.subplots_adjust(top=0.90, bottom=0.15, left=0.15)
ax1 = fig.add_subplot(111)
plt.plot(tx_all.phi, tx_all.g_all_snow_off, label="snow off")
# plt.plot(90 - tx_all.phi, tx_all.g_tx_mean_n_snow_off, label="Snow off north")
# plt.plot(90 - tx_all.phi, tx_all.g_tx_mean_e_snow_off, label="Snow off east")
# plt.plot(90 - tx_all.phi, tx_all.g_tx_mean_s_snow_off, label="Snow off south")
# plt.plot(90 - tx_all.phi, tx_all.g_tx_mean_w_snow_off, label="Snow off west")
plt.plot(tx_all.phi, tx_all.g_all_snow_on, label="snow on")
# plt.scatter(bin_mid, g_hemi, label="hemi")
# plt.plot(90 - tx_all.phi, tx_all.g_tx_mean_n_snow_on, label="Snow on north")
# plt.plot(90 - tx_all.phi, tx_all.g_tx_mean_e_snow_on, label="Snow on east")
# plt.plot(90 - tx_all.phi, tx_all.g_tx_mean_s_snow_on, label="Snow on south")
# plt.plot(90 - tx_all.phi, tx_all.g_tx_mean_w_snow_on, label="Snow on west")
# plt.plot(tx_all.phi, gx_0, label="gx_0")
# plt.plot(tx_all.phi, gx_h5, label="gx_0.5")
# plt.plot(tx_all.phi, gx_h75, label="gx_0.75")
# # plt.plot(tx_all.phi, gx_1, label="gx_1")
# plt.plot(tx_all.phi, gx_inf, label="gx_inf")
plt.xlim([0, 90])
plt.ylim([0, 1])
plt.legend()
ax1.set_ylabel("Extinction efficiency [-]")
ax1.set_xlabel("Elevation angle [$^{\circ}$]")
fig.savefig(plot_out_dir + "uf_g.png")




fig = plt.figure()
fig.subplots_adjust(top=0.90, bottom=0.15, left=0.15)
ax1 = fig.add_subplot(111)
plt.plot(tx_all.phi, tx_all.tx_mean_all_snow_off, label="snow off")
off_lower_bound = tx_all.tx_mean_all_snow_off - tx_all.tx_sd_all_snow_off
off_upper_bound = tx_all.tx_mean_all_snow_off + tx_all.tx_sd_all_snow_off
plt.fill_between(bins, off_lower_bound, off_upper_bound, alpha=0.5)

plt.plot(tx_all.phi, tx_all.tx_mean_all_snow_on, label="snow on")
on_lower_bound = tx_all.tx_mean_all_snow_on - tx_all.tx_sd_all_snow_on
on_upper_bound = tx_all.tx_mean_all_snow_on + tx_all.tx_sd_all_snow_on
plt.fill_between(bins, on_lower_bound, on_upper_bound, alpha=0.5)

plt.xlim(0, 90)
plt.ylim(0, 1)
plt.legend()
ax1.set_ylabel("Light transmittance [-]")
ax1.set_xlabel("Angle from zenith [$^{\circ}$]")
fig.savefig(plot_out_dir + "uf_tx_zenith_all.png")

# LAS classes:
# 1 -- unclassified
# 2 -- ground
# 5 -- vegetation
# 7 -- ground noise
# 8 -- vegetation noise
#
# ## old hat below
#
# # # filter by scanning pattern
# # grid60 = (las_data.gps_time >= 324565) & (las_data.gps_time <= 324955)  # 20% of data
# # grid120 = (las_data.gps_time >= 325588) & (las_data.gps_time <= 325800)  # 6% of data
# # f_ladder = (las_data.gps_time >= 326992) & (las_data.gps_time <= 327260)  # 16% of data
# # f_spin = (las_data.gps_time >= 325018) & (las_data.gps_time <= 325102)  # 18% of data
# #
# # las_data = las_data[grid120 | grid60 | f_ladder | f_spin]
# #
# # np.sum(f_spin)/las_data.__len__()
#
# las_data = las_data.assign(afn_bin=las_data.angle_from_nadir_deg.astype(int))
#
# # count by classification, first/last, and scan_angle
# FG = las_data[(las_data.return_num == 1) & (las_data.classification == 2)].groupby('afn_bin').size()
# FC = las_data[(las_data.return_num == 1) & (las_data.classification == 5)].groupby('afn_bin').size()
# LG = las_data[(las_data.return_num == las_data.num_returns) & (las_data.classification == 2)].groupby('afn_bin').size()
# LC = las_data[(las_data.return_num == las_data.num_returns) & (las_data.classification == 5)].groupby('afn_bin').size()
# SG = las_data[(las_data.num_returns == 1) & (las_data.classification == 2)].groupby('afn_bin').size()
# SC = las_data[(las_data.num_returns == 1) & (las_data.classification == 5)].groupby('afn_bin').size()
# # concatinate
# lpm = pd.concat([FG, FC, LG, LC, SG, SC], ignore_index=False, axis=1)
# lpm.columns = ['FG', 'FC', 'LG', 'LC', 'SG', 'SC']
#
# # want to know, for each sample, at each angle, what is the probability of returning a certain class
# n_returns = las_data.groupby('afn_bin').size()
# n_samples = las_data[(las_data.return_num == 1)].groupby('afn_bin').size()
# n_samples_cum = np.nancumsum(n_samples)
# nn = las_data.gps_time.nunique()
#
# # plot counts
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
#
#
# plt.plot(n_samples.index, n_samples)
# plt.plot(n_samples.index, n_samples_cum/nn)
#
# fig, ax = plt.subplots()
# ax.plot(lpm.index, lpm.FG/n_samples, label="first ground")
# ax.plot(lpm.index, lpm.FC/n_samples, label="first canopy")
# ax.plot(lpm.index, lpm.LG/n_samples, label="last ground")
# ax.plot(lpm.index, lpm.LC/n_samples, label="last canopy")
# # plt.plot(lpm.index, lpm.SG/n_samples, label="single ground")  # same as first return ground FG
# # plt.plot(lpm.index, lpm.SC/n_samples, label="single canopy")
# ax.set(xlabel="scan angle (deg from nadir)", ylabel="relative frequency per sample", title="Frequency of returns by scan angle")
# ax.grid()
# ax.legend(title="return classification")
# plt.show()
#
# # calculate lpms -- 1 if all ground, 0 if all canopy
# lpmf = lpm.FG/(lpm.FG + lpm.FC)
# lpml = (lpm.FG + lpm.LG)/(lpm.FG + lpm.LG + lpm.FC)
# lpmc = lpm.LG/(lpm.LG + lpm.FC)
#
# # calculation of LMP_all
# first_returns = las_data[las_data.return_num == 1]
# first_returns = first_returns.assign(num_returns_inv=1/first_returns.num_returns)
# first_returns = first_returns.groupby("afn_bin")
# sum_inv_ret = first_returns.num_returns_inv.sum()
#
# lpma = 1 - sum_inv_ret/n_samples
#
# # path length correction... assume 1/cosine (beer-lambert)
# plc = 1/np.cos(lpm.index*np.pi/180)
#
# # plot lpms
# plt.plot(lpmf.index, lpmf.values, label="LPM_firsts")
# plt.plot(lpml.index, lpml.values, label="LPM_lasts")
# plt.plot(lpmc.index, lpmc.values, label="LPM_canopy")
# plt.plot(lpma.index, lpma.values, label="LPM_all")
#
# # path-corrected lpms
# fig, ax = plt.subplots()
# ax.plot(lpmf.index, lpmf.values * plc, label="LPM_firsts")
# ax.plot(lpml.index, lpml.values * plc, label="LPM_lasts")
# ax.plot(lpmc.index, lpmc.values * plc, label="LPM_canopy")
# # plt.plot(lpma.index, lpma.values * plc, label="LPM_all")  # not appropriate in this analysis
# ax.set(xlabel="scan angle (deg from nadir)", ylabel="LPM value", title="Three laser penetration metics (LPMs) by scan angle")
# ax.legend()
# ax.grid()
# plt.show()
#
# # how do we account for clumping?
# import pdal
#
# las_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\LAS\\19_149_UF.las"
#
# json = """
# [
#     "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/19_149/19_149_snow_off/OUTPUT_FILES/LAS/19_149_UF.las",
#     {
#         "type": "filters.sample",
#         "radius": "0.15"
#     },
#     {
#         "type": "writers.las",
#         "filename": "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/19_149/19_149_snow_off/OUTPUT_FILES/LAS/19_149_UF_sampled.las"
#     }
# ]
# """
#
# pipeline = pdal.Pipeline(json)
# count = pipeline.execute()
# arrays = pipeline.arrays
# metadata = pipeline.metadata
# log = pipeline.log
