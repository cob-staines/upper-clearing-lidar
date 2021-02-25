####
import geotk as gt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

plot_out_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\graphics\\thesis_graphics\\frequency distributions\\"

data_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\uf_merged_.05m_ahpl_native.csv'
data = pd.read_csv(data_in)
# filter to upper forest
data = data.loc[data.uf == 1, :]

# convert from returns to contact number
data.loc[:, "cn"] = data.er_p0_mean * 0.19447
# calculate transmission (spherical leaf angle assumption)
data.loc[:, "trans_rs"] = np.exp(-data.cn)

def resampling_histoplot(data, proposal, sample, nbins):
    d_samp, stats = gt.rejection_sample(data, proposal, sample, nbins, original_df=False)
    set_a = data.assign(set="observed")
    set_b = d_samp.assign(set="bias corrected")
    ab = pd.concat([set_a.loc[:, [proposal, "set"]], set_b.loc[:, [proposal, "set"]]])
    plot = sns.histplot(ab, x=proposal, hue="set", stat="density", common_norm=False, element="step")
    return plot, d_samp


#### rejection sample swe
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of SWE for 14 Feb. 2019\n Upper Forest, 5cm resolution, bias corrected with $LAI_{rs}$')
ax1.set_xlabel("SWE [mm]")
ax1.set_ylabel("Relative frequency [-]")
g, d_045 = resampling_histoplot(data, 'swe_19_045', 'cn', 50)
g.legend_.set_title(None)
fig.savefig(plot_out_dir + "freq_dist_resampled_swe_045.png")

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of SWE for 19 Feb. 2019\n Upper Forest, 5cm resolution, bias corrected with $LAI_{rs}$')
ax1.set_xlabel("SWE [mm]")
ax1.set_ylabel("Relative frequency [-]")
g, d_050 = resampling_histoplot(data, 'swe_19_050', 'cn', 50)
g.legend_.set_title(None)
fig.savefig(plot_out_dir + "freq_dist_resampled_swe_050.png")

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of SWE for 21 Feb. 2019\n Upper Forest, 5cm resolution, bias corrected with $LAI_{rs}$')
ax1.set_xlabel("SWE [mm]")
ax1.set_ylabel("Relative frequency [-]")
g, d_052 = resampling_histoplot(data, 'swe_19_052', 'cn', 50)
g.legend_.set_title(None)
fig.savefig(plot_out_dir + "freq_dist_resampled_swe_052.png")

# plot all together
d_045 = d_045.assign(date="045", swe=d_045.swe_19_045)
d_050 = d_050.assign(date="050", swe=d_050.swe_19_050)
d_052 = d_052.assign(date="052", swe=d_052.swe_19_052)
all_swe = pd.concat([d_045.loc[:, ["swe", "date"]], d_050.loc[:, ["swe", "date"]], d_052.loc[:, ["swe", "date"]]])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distributions of SWE for all days\n Upper Forest, 5cm resolution, bias corrected with $LAI_{rs}$')
ax1.set_xlabel("SWE [mm]")
ax1.set_ylabel("Relative frequency [-]")
plot = sns.histplot(all_swe, x="swe", hue="date", stat="density", common_norm=False, element="step")
fig.savefig(plot_out_dir + "freq_dist_resampled_all_swe.png")

# export rejection sampled points
e_045 = d_045.loc[:, ["x_coord", "y_coord", "swe_19_045"]]
e_050 = d_050.loc[:, ["x_coord", "y_coord", "swe_19_050"]]
e_052 = d_052.loc[:, ["x_coord", "y_coord", "swe_19_052"]]

rej_samp_out_file = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\rejection_sampled\\'
e_045.to_csv(rej_samp_out_file + 'resampled_swe_19_045_ahpl_r.05m_interp2x_by_contact-number.csv', index=False)
e_050.to_csv(rej_samp_out_file + 'resampled_swe_19_050_ahpl_r.05m_interp2x_by_contact-number.csv', index=False)
e_052.to_csv(rej_samp_out_file + 'resampled_swe_19_052_ahpl_r.05m_interp2x_by_contact-number.csv', index=False)


#### rejection sample dswe
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of $\Delta$SWE for 14-19 Feb. 2019\n Upper Forest, 5cm resolution, bias corrected with $LAI_{rs}$')
ax1.set_xlabel("SWE [mm]")
ax1.set_ylabel("Relative frequency [-]")
g, d_045_050 = resampling_histoplot(data, 'dswe_19_045-19_050', 'cn', 50)
g.legend_.set_title(None)
fig.savefig(plot_out_dir + "freq_dist_resampled_dswe_045-050.png")

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of $\Delta$SWE for 19-21 Feb. 2019\n Upper Forest, 5cm resolution, bias corrected with $LAI_{rs}$')
ax1.set_xlabel("SWE [mm]")
ax1.set_ylabel("Relative frequency [-]")
g, d_050_052 = resampling_histoplot(data, 'dswe_19_050-19_052', 'cn', 50)
g.legend_.set_title(None)
fig.savefig(plot_out_dir + "freq_dist_resampled_dswe_050-052.png")

# plot all together
d_045_050 = d_045_050.assign(interval="045-050", dswe=d_045_050.loc[:, "dswe_19_045-19_050"])
d_050_052 = d_050_052.assign(interval="050-052", dswe=d_050_052.loc[:, "dswe_19_050-19_052"])
all_dswe = pd.concat([d_045_050.loc[:, ["dswe", "interval"]], d_050_052.loc[:, ["dswe", "interval"]]])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distributions of $\Delta$SWE for all days\n Upper Forest, 5cm resolution, bias corrected with $LAI_{rs}$')
ax1.set_xlabel("$\Delta$SWE [mm]")
ax1.set_ylabel("Relative frequency [-]")
sns.histplot(all_dswe, x="dswe", hue="interval", stat="density", common_norm=False, element="step")
fig.savefig(plot_out_dir + "freq_dist_resampled_all_dswe.png")


##### lai and transmittance
data_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\mb_15_merged_.25m_native_canopy_19_149.csv'
c_data = pd.read_csv(data_in)
# filter to upper forest
c_data = c_data.loc[c_data.uf == 1, :]
# convert from returns to contact number
c_data.loc[:, "cn"] = c_data.er_p0_mean * 0.19447
# calculate transmission (spherical leaf angle assumption)
c_data.loc[:, "trans_rs"] = np.exp(-c_data.cn)

# # seperate plots
# sns.histplot(c_data, x="lai_s_cc", stat="density", element="step")
# sns.histplot(c_data, x="cn", stat="density", element="step")

# plot LAI against one another
set_a = c_data.assign(method="Ray Sampling", lai=c_data.cn)
set_b = c_data.assign(method="Hemi-photo 15deg", lai=c_data.contactnum_1)
set_c = c_data.assign(method="Hemi-photo 75deg", lai=c_data.lai_s_cc)
# set_b = c_data.assign(method="Hemispherical", lai=c_data.lai_s_cc)
ab = pd.concat([set_a.loc[:, ["lai", "method"]], set_b.loc[:, ["lai", "method"]], set_c.loc[:, ["lai", "method"]]])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distributions of LAI\n Upper Forest, 25cm resolution, snow-free canopy')
ax1.set_xlabel("LAI [-]")
ax1.set_ylabel("Relative frequency [-]")
sns.histplot(ab, x="lai", hue="method", stat="density", common_norm=False, element="step")
plt.xlim(0, 10)
fig.savefig(plot_out_dir + "freq_dist_lai_uf.png")

# # plot transmittance against one another
# sns.histplot(c_data, x="trans_rs", stat="density", element="step", bins=30)
# sns.histplot(c_data, x="transmission", stat="density", element="step", bins=30)
# sns.histplot(c_data, x="transmission_1", stat="density", element="step", bins=30)

set_a = c_data.assign(method="Ray Sampling", trans=c_data.trans_rs)
set_b = c_data.assign(method="Hemi-photo 15deg", trans=c_data.transmission_s_1)
set_c = c_data.assign(method="Hemi-photo 75deg", trans=c_data.transmission_gaps)
ab = pd.concat([set_a.loc[:, ["trans", "method"]], set_b.loc[:, ["trans", "method"]],  set_c.loc[:, ["trans", "method"]]])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distributions of light transmittance\n Upper Forest, 25cm resolution, snow-free canopy')
ax1.set_xlabel("Transmittance [-]")
ax1.set_ylabel("Relative frequency [-]")
sns.histplot(ab, x="trans", hue="method", stat="density", common_norm=False, element="step")
fig.savefig(plot_out_dir + "freq_dist_trans_uf.png")


### other canopy metrics
data_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\mb_15_merged_.10m_native_canopy_19_149.csv'
cc_data = pd.read_csv(data_in)
# filter to upper forest
cc_data = cc_data.loc[cc_data.uf == 1, :]

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of Distance to Nearest Tree (DNT)\n Upper Forest, 10cm resolution, snow-free canopy')
ax1.set_xlabel("DNT [m]")
ax1.set_ylabel("Relative frequency [-]")
sns.histplot(cc_data, x="dnt", stat="density", element="step")
fig.savefig(plot_out_dir + "freq_dist_dnt.png")

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of Distance from Canopy Edge (DCE)\n Upper Forest, 10cm resolution, snow-free canopy')
ax1.set_xlabel("DCE [m]")
ax1.set_ylabel("Relative frequency [-]")
sns.histplot(cc_data, x="dce", stat="density", element="step", binwidth=0.1)
fig.savefig(plot_out_dir + "freq_dist_dce.png")

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of vegetation crown height\n Upper Forest, 10cm resolution, snow-free vegetation > 1m')
ax1.set_xlabel("Vegetation crown height [m]")
ax1.set_ylabel("Relative frequency [-]")
sns.histplot(cc_data.loc[cc_data.chm > 1, ], x="chm", stat="density", element="step")
fig.savefig(plot_out_dir + "freq_dist_chm.png")



set_a = cc_data.assign(method="First", lpm=cc_data.lpmf15)
set_b = cc_data.assign(method="Last", lpm=cc_data.lpml15)
set_c = cc_data.assign(method="Canopy", lpm=cc_data.lpmc15)
ab = pd.concat([set_a.loc[:, ["lpm", "method"]], set_b.loc[:, ["lpm", "method"]],  set_c.loc[:, ["lpm", "method"]]])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distributions of light transmittance\n Upper Forest, 25cm resolution, snow-free canopy')
ax1.set_xlabel("Transmittance [-]")
ax1.set_ylabel("Relative frequency [-]")
sns.histplot(ab, x="lpm", hue="method", stat="density", common_norm=False, element="step", bins=20)
fig.savefig(plot_out_dir + "freq_dist_trans_uf.png")
