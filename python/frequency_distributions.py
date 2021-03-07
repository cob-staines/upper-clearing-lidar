####
import geotk as gt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

plot_out_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\graphics\\thesis_graphics\\frequency distributions\\"
rej_samp_out_file = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\rejection_sampled\\'

data_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\uf_merged_.05m_ahpl_native.csv'
data = pd.read_csv(data_in)
# convert from returns to contact number
data.loc[:, "cn"] = data.er_p0_mean * 0.19447
# calculate transmission (spherical leaf angle assumption)
data.loc[:, "trans_rs"] = np.exp(-data.cn)

# filter to upper forest
data_uf = data.loc[data.plots == 1, :]
data_uc = data.loc[data.plots == 2, :]


data_10_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\uf_merged_.10m_ahpl_median_canopy_19_149.csv'
data_10 = pd.read_csv(data_10_in)

# filter to upper forest
data_10_uf = data_10.loc[data_10.plots == 1, :]
data_10_uc = data_10.loc[data_10.plots == 2, :]

def resampling_histoplot(data, proposal, sample, nbins):
    d_samp, stats = gt.rejection_sample(data, proposal, sample, nbins, original_df=False)
    set_a = data.assign(set="observed")
    set_b = d_samp.assign(set="bias-corrected")
    ab = pd.concat([set_a.loc[:, [proposal, "set"]], set_b.loc[:, [proposal, "set"]]])
    plot = sns.histplot(ab, x=proposal, hue="set", stat="density", common_norm=False, element="step")
    return plot, d_samp

# first, some plots of different canopy metrics to assess which would be best for bias correction


def proposal_comparison_histplot(data, proposal, sample, bins="auto"):
    prop = data.loc[:, [sample]]
    prop = prop.assign(tag="all")
    samp = data.loc[~np.isnan(data.loc[:, proposal]), [sample]]
    samp = samp.assign(tag="sampled")
    ps = pd.concat([prop, samp])

    plot = sns.histplot(ps, x=sample, bins=bins, hue="tag", stat="density", common_norm=False, element="step", hue_order=["sampled", "all"])
    return plot


# sample bias plots
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Ground sample bias of LPM-Last for 21 Feb. 2019\n Upper Forest, 10cm resolution')
ax1.set_xlabel("LPM-Last [-]")
ax1.set_ylabel("Relative frequency [-]")
g = proposal_comparison_histplot(data_uf, 'swe_19_052', 'lpml15', bins=30)
g.legend_.set_title(None)
legend = g.get_legend()
handles = legend.legendHandles
legend.remove()
g.legend(handles, ["SWE only", "all"], loc="upper center")
fig.savefig(plot_out_dir + "sample_bias_lpml15_with_swe_19_052_uf.png")

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Ground sample bias of LPM-Last for 21 Feb. 2019\n Upper Clearing, 10cm resolution')
ax1.set_xlabel("LPM-Last [-]")
ax1.set_ylabel("Relative frequency [-]")
g = proposal_comparison_histplot(data_uc, 'swe_19_052', 'lpml15', bins=30)
g.legend_.set_title(None)
legend = g.get_legend()
handles = legend.legendHandles
legend.remove()
g.legend(handles, ["SWE only", "all"], loc="upper center")
fig.savefig(plot_out_dir + "sample_bias_lpml15_with_swe_19_052_uc.png")



### uf ###

#### rejection sample swe
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of SWE for 14 Feb. 2019\n Upper Forest, 5cm resolution, bias corrected with LPM-Last')
ax1.set_xlabel("SWE [mm]")
ax1.set_ylabel("Relative frequency [-]")
g, d_045_uf = resampling_histoplot(data_uf, 'swe_19_045', 'lpml15', 50)
g.legend_.set_title(None)
fig.savefig(plot_out_dir + "freq_dist_resampled_swe_045_uf_lpml15.png")

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of SWE for 19 Feb. 2019\n Upper Forest, 5cm resolution, bias corrected with LPM-Last')
ax1.set_xlabel("SWE [mm]")
ax1.set_ylabel("Relative frequency [-]")
g, d_050_uf = resampling_histoplot(data_uf, 'swe_19_050', 'lpml15', 50)
g.legend_.set_title(None)
fig.savefig(plot_out_dir + "freq_dist_resampled_swe_050_uf_lpml15.png")

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of SWE for 21 Feb. 2019\n Upper Forest, 5cm resolution, bias corrected with LPM-Last')
ax1.set_xlabel("SWE [mm]")
ax1.set_ylabel("Relative frequency [-]")
g, d_052_uf = resampling_histoplot(data_uf, 'swe_19_052', 'lpml15', 50)
g.legend_.set_title(None)
fig.savefig(plot_out_dir + "freq_dist_resampled_swe_052_uf_lpml15.png")

# plot all together
d_045_uf = d_045_uf.assign(date="14 Feb", swe=d_045_uf.swe_19_045)
d_050_uf = d_050_uf.assign(date="19 Feb", swe=d_050_uf.swe_19_050)
d_052_uf = d_052_uf.assign(date="21 Feb", swe=d_052_uf.swe_19_052)
all_swe_uf = pd.concat([d_045_uf.loc[:, ["swe", "date"]], d_050_uf.loc[:, ["swe", "date"]], d_052_uf.loc[:, ["swe", "date"]]])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distributions of SWE for all days\n Upper Forest, 5cm resolution, bias corrected with LPM-Last')
ax1.set_xlabel("SWE [mm]")
ax1.set_ylabel("Relative frequency [-]")
plot = sns.histplot(all_swe_uf, x="swe", hue="date", stat="density", common_norm=False, element="step")
fig.savefig(plot_out_dir + "freq_dist_resampled_all_swe_uf_lpml15.png")

# export rejection sampled points
e_045_uf = d_045_uf.loc[:, ["x_coord", "y_coord", "swe_19_045"]]
e_050_uf = d_050_uf.loc[:, ["x_coord", "y_coord", "swe_19_050"]]
e_052_uf = d_052_uf.loc[:, ["x_coord", "y_coord", "swe_19_052"]]

e_045_uf.to_csv(rej_samp_out_file + 'resampled_swe_19_045_uf_ahpl_r.05m_interp2x_by_lpml15.csv', index=False)
e_050_uf.to_csv(rej_samp_out_file + 'resampled_swe_19_050_uf_ahpl_r.05m_interp2x_by_lpml15.csv', index=False)
e_052_uf.to_csv(rej_samp_out_file + 'resampled_swe_19_052_uf_ahpl_r.05m_interp2x_by_lpml15.csv', index=False)


#### rejection sample dswe
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of $\Delta$SWE for 14-19 Feb. 2019\n Upper Forest, 5cm resolution, bias corrected with LPM-Last')
ax1.set_xlabel("SWE [mm]")
ax1.set_ylabel("Relative frequency [-]")
g, d_045_050_uf = resampling_histoplot(data_uf, 'dswe_19_045-19_050', 'lpml15', 50)
g.legend_.set_title(None)
fig.savefig(plot_out_dir + "freq_dist_resampled_dswe_045-050_uf_lpml15.png")

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of $\Delta$SWE for 19-21 Feb. 2019\n Upper Forest, 5cm resolution, bias corrected with LPM-Last')
ax1.set_xlabel("SWE [mm]")
ax1.set_ylabel("Relative frequency [-]")
g, d_050_052_uf = resampling_histoplot(data_uf, 'dswe_19_050-19_052', 'lpml15', 50)
g.legend_.set_title(None)
fig.savefig(plot_out_dir + "freq_dist_resampled_dswe_050-052_uf_lpml15.png")

# plot all together
d_045_050_uf = d_045_050_uf.assign(interval="14-19 Feb", dswe=d_045_050_uf.loc[:, "dswe_19_045-19_050"])
d_050_052_uf = d_050_052_uf.assign(interval="19-21 Feb", dswe=d_050_052_uf.loc[:, "dswe_19_050-19_052"])
all_dswe_uf = pd.concat([d_045_050_uf.loc[:, ["dswe", "interval"]], d_050_052_uf.loc[:, ["dswe", "interval"]]])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distributions of $\Delta$SWE for all days\n Upper Forest, 5cm resolution, bias corrected with LPM-last')
ax1.set_xlabel("$\Delta$SWE [mm]")
ax1.set_ylabel("Relative frequency [-]")
sns.histplot(all_dswe_uf, x="dswe", hue="interval", stat="density", common_norm=False, element="step")
fig.savefig(plot_out_dir + "freq_dist_resampled_all_dswe_uf_lpml15.png")

# export rejection sampled points
e_045_050_uf = d_045_050_uf.loc[:, ["x_coord", "y_coord", "dswe_19_045-19_050"]]
e_050_052_uf = d_050_052_uf.loc[:, ["x_coord", "y_coord", "dswe_19_050-19_052"]]

e_045_050_uf.to_csv(rej_samp_out_file + 'resampled_dswe_19_045-19_050_uf_ahpl_r.05m_interp2x_by_lpml15.csv', index=False)
e_050_052_uf.to_csv(rej_samp_out_file + 'resampled_dswe_19_050-19_052_uf_ahpl_r.05m_interp2x_by_lpml15.csv', index=False)


### uc ###

#### rejection sample swe
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of SWE for 14 Feb. 2019\n Upper Clearing, 5cm resolution, bias corrected with LPM-Last')
ax1.set_xlabel("SWE [mm]")
ax1.set_ylabel("Relative frequency [-]")
g, d_045_uc = resampling_histoplot(data_uc, 'swe_19_045', 'lpml15', 50)
g.legend_.set_title(None)
fig.savefig(plot_out_dir + "freq_dist_resampled_swe_045_lpml15.png")

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of SWE for 19 Feb. 2019\n Upper Clearing, 5cm resolution, bias corrected with LPM-Last')
ax1.set_xlabel("SWE [mm]")
ax1.set_ylabel("Relative frequency [-]")
g, d_050_uc = resampling_histoplot(data_uc, 'swe_19_050', 'lpml15', 50)
g.legend_.set_title(None)
fig.savefig(plot_out_dir + "freq_dist_resampled_swe_050_lpml15.png")

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of SWE for 21 Feb. 2019\n Upper Clearing, 5cm resolution, bias corrected with LPM-Last')
ax1.set_xlabel("SWE [mm]")
ax1.set_ylabel("Relative frequency [-]")
g, d_052_uc = resampling_histoplot(data_uc, 'swe_19_052', 'lpml15', 50)
g.legend_.set_title(None)
fig.savefig(plot_out_dir + "freq_dist_resampled_swe_052_uc_lpml15.png")

# plot all together
d_045_uc = d_045_uc.assign(date="14 Feb", swe=d_045_uc.swe_19_045)
d_050_uc = d_050_uc.assign(date="19 Feb", swe=d_050_uc.swe_19_050)
d_052_uc = d_052_uc.assign(date="21 Feb", swe=d_052_uc.swe_19_052)
all_swe_uc = pd.concat([d_045_uc.loc[:, ["swe", "date"]], d_050_uc.loc[:, ["swe", "date"]], d_052_uc.loc[:, ["swe", "date"]]])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distributions of SWE for all days\n Upper Clearing, 5cm resolution, bias corrected with LPM-Last')
ax1.set_xlabel("SWE [mm]")
ax1.set_ylabel("Relative frequency [-]")
plot = sns.histplot(all_swe_uc, x="swe", hue="date", stat="density", common_norm=False, element="step")
fig.savefig(plot_out_dir + "freq_dist_resampled_all_swe_uc_lpmf15.png")

# export rejection sampled points
e_045_uc = d_045_uc.loc[:, ["x_coord", "y_coord", "swe_19_045"]]
e_050_uc = d_050_uc.loc[:, ["x_coord", "y_coord", "swe_19_050"]]
e_052_uc = d_052_uc.loc[:, ["x_coord", "y_coord", "swe_19_052"]]

e_045_uc.to_csv(rej_samp_out_file + 'resampled_swe_19_045_uc_ahpl_r.05m_interp2x_by_lpml15.csv', index=False)
e_050_uc.to_csv(rej_samp_out_file + 'resampled_swe_19_050_uc_ahpl_r.05m_interp2x_by_lpml15.csv', index=False)
e_052_uc.to_csv(rej_samp_out_file + 'resampled_swe_19_052_uc_ahpl_r.05m_interp2x_by_lpml15.csv', index=False)


#### rejection sample dswe
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of $\Delta$SWE for 14-19 Feb. 2019\n Upper Clearing, 5cm resolution, bias corrected with LPM-Last')
ax1.set_xlabel("SWE [mm]")
ax1.set_ylabel("Relative frequency [-]")
g, d_045_050_uc = resampling_histoplot(data_uc, 'dswe_19_045-19_050', 'lpml15', 50)
g.legend_.set_title(None)
fig.savefig(plot_out_dir + "freq_dist_resampled_dswe_045-050_uc_lpml15.png")

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of $\Delta$SWE for 19-21 Feb. 2019\n Upper Clearing, 5cm resolution, bias corrected with LPM-Last')
ax1.set_xlabel("SWE [mm]")
ax1.set_ylabel("Relative frequency [-]")
g, d_050_052_uc = resampling_histoplot(data_uc, 'dswe_19_050-19_052', 'lpml15', 50)
g.legend_.set_title(None)
fig.savefig(plot_out_dir + "freq_dist_resampled_dswe_050-052_uc_lpml15.png")

# plot all together
d_045_050_uc = d_045_050_uc.assign(interval="14-19 Feb", dswe=d_045_050_uc.loc[:, "dswe_19_045-19_050"])
d_050_052_uc = d_050_052_uc.assign(interval="19-21 Feb", dswe=d_050_052_uc.loc[:, "dswe_19_050-19_052"])
all_dswe_uc = pd.concat([d_045_050_uc.loc[:, ["dswe", "interval"]], d_050_052_uc.loc[:, ["dswe", "interval"]]])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distributions of $\Delta$SWE for all days\n Upper Clearing, 5cm resolution, bias corrected with LPM-Last')
ax1.set_xlabel("$\Delta$SWE [mm]")
ax1.set_ylabel("Relative frequency [-]")
sns.histplot(all_dswe_uc, x="dswe", hue="interval", stat="density", common_norm=False, element="step")
fig.savefig(plot_out_dir + "freq_dist_resampled_all_dswe_uc_lpml15.png")

# export rejection sampled points
e_045_050_uc = d_045_050_uc.loc[:, ["x_coord", "y_coord", "dswe_19_045-19_050"]]
e_050_052_uc = d_050_052_uc.loc[:, ["x_coord", "y_coord", "dswe_19_050-19_052"]]

e_045_050_uc.to_csv(rej_samp_out_file + 'resampled_dswe_19_045-19_050_uc_ahpl_r.05m_interp2x_by_lpml15.csv', index=False)
e_050_052_uc.to_csv(rej_samp_out_file + 'resampled_dswe_19_050-19_052_uc_ahpl_r.05m_interp2x_by_lpml15.csv', index=False)

# combined plot plots
# plot 045-050 all together
d_045_050_uf = d_045_050_uf.assign(plot="upper forest", dswe=d_045_050_uf.loc[:, "dswe_19_045-19_050"])
d_045_050_uc = d_045_050_uc.assign(plot="upper clearing", dswe=d_045_050_uc.loc[:, "dswe_19_045-19_050"])
d_045_050_dswe = pd.concat([d_045_050_uf.loc[:, ["dswe", "plot"]], d_045_050_uc.loc[:, ["dswe", "plot"]]])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distributions of $\Delta$SWE for 14-19 Feb. 2019\n 5cm resolution, bias corrected with LPM-Last')
ax1.set_xlabel("$\Delta$SWE [mm]")
ax1.set_ylabel("Relative frequency [-]")
sns.histplot(d_045_050_dswe, x="dswe", hue="plot", stat="density", common_norm=False, element="step")
fig.savefig(plot_out_dir + "freq_dist_resampled_045_050_dswe_lpml15.png")

# plot 050-052 all together
d_050_052_uf = d_050_052_uf.assign(plot="upper forest", dswe=d_050_052_uf.loc[:, "dswe_19_050-19_052"])
d_050_052_uc = d_050_052_uc.assign(plot="upper clearing", dswe=d_050_052_uc.loc[:, "dswe_19_050-19_052"])
d_050_052_dswe = pd.concat([d_050_052_uf.loc[:, ["dswe", "plot"]], d_050_052_uc.loc[:, ["dswe", "plot"]]])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distributions of $\Delta$SWE for 19-21 Feb. 2019\n 5cm resolution, bias corrected with LPM-Last')
ax1.set_xlabel("$\Delta$SWE [mm]")
ax1.set_ylabel("Relative frequency [-]")
sns.histplot(d_050_052_dswe, x="dswe", hue="plot", stat="density", common_norm=False, element="step")
fig.savefig(plot_out_dir + "freq_dist_resampled_050_052_dswe_lpml15.png")

##### lai and transmittance
data_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\mb_15_merged_.25m_native_canopy_19_149.csv'
c_data = pd.read_csv(data_in)
# filter to upper forest
c_data = c_data.loc[c_data.plots == 1, :]  # uf only
# convert from returns to contact number
c_data.loc[:, "cn"] = c_data.er_p0_mean * 0.19447
# calculate transmission (spherical leaf angle assumption)
c_data.loc[:, "trans_rs"] = np.exp(-c_data.cn)

# # seperate plots
# sns.histplot(c_data, x="lai_s_cc", stat="density", element="step")
# sns.histplot(c_data, x="cn", stat="density", element="step")

# plot LAI against one another
set_a = c_data.assign(method="Ray Sampling 1deg", lai=c_data.cn)
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

set_a = c_data.assign(method="Ray Sampling 1deg", trans=c_data.trans_rs)
set_b = c_data.assign(method="Hemi-photo 15deg", trans=c_data.transmission_s_1)
set_c = c_data.assign(method="Hemi-photo 75deg", trans=c_data.transmission_gaps)
ab = pd.concat([set_a.loc[:, ["trans", "method"]], set_b.loc[:, ["trans", "method"]],  set_c.loc[:, ["trans", "method"]]])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distributions of light transmittance\n Upper Forest, 25cm resolution, snow-free canopy')
ax1.set_xlabel("Transmittance [-]")
ax1.set_ylabel("Relative frequency [-]")
sns.histplot(ab, x="trans", hue="method", stat="density", common_norm=False, element="step", bins=70)
fig.savefig(plot_out_dir + "freq_dist_trans_uf.png")


### other canopy metrics
data_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\mb_15_merged_.10m_native_canopy_19_149.csv'
cc_data = pd.read_csv(data_in)
# filter to upper forest
cc_data = cc_data.loc[cc_data.plots == 1, :]  # uf only

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of Distance to Nearest Tree (DNT)\n Upper Forest, 10cm resolution, snow-free canopy')
ax1.set_xlabel("DNT [m]")
ax1.set_ylabel("Relative frequency [-]")
sns.histplot(cc_data, x="dnt", stat="density", element="step")
fig.savefig(plot_out_dir + "freq_dist_dnt_uf.png")

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of Distance from Canopy Edge (DCE)\n Upper Forest, 10cm resolution, snow-free canopy')
ax1.set_xlabel("DCE [m]")
ax1.set_ylabel("Relative frequency [-]")
sns.histplot(cc_data, x="dce", stat="density", element="step", binwidth=0.1)
fig.savefig(plot_out_dir + "freq_dist_dce_uf.png")

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of vegetation crown height\n Upper Forest, 10cm resolution, snow-free vegetation > 1m')
ax1.set_xlabel("Vegetation crown height [m]")
ax1.set_ylabel("Relative frequency [-]")
sns.histplot(cc_data.loc[cc_data.chm > 1, ], x="chm", stat="density", element="step")
fig.savefig(plot_out_dir + "freq_dist_chm_uf.png")



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
