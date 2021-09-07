# plots of frequency distributions of snow depth, SWE, dSWE products over upper forest and upper clearing plots
# includes rejection sampling methods for areal statistics and distributions

import libraries.geotk as gt
import libraries.raslib
import matplotlib
# matplotlib.use("Agg")
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

plot_out_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\graphics\\thesis_graphics\\frequency distributions\\"
rej_samp_out_file = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\rejection_sampled\\'

# upper forest snow
data_uf_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\merged_data_products\\merged_uf_.05m_snow_nearest_canopy_19_149.csv'
data_uf = pd.read_csv(data_uf_in)

# upper clearing snow
data_uc_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\merged_data_products\\merged_uc_.05m_snow_nearest_canopy_19_149.csv'
data_uc = pd.read_csv(data_uc_in)

# # upper forest canopy
# data_10_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\uf_merged_.10m_ahpl_median_canopy_19_149.csv'
# data_10 = pd.read_csv(data_10_in)
#
# filter to upper forest
# data_10_uf = data_10.loc[data_10.plots == 1, :]
# data_10_uc = data_10.loc[data_10.plots == 2, :]


def resampling_histoplot(data, proposal, sample, nbins, plotbins='auto', extra_col=None, extra_lab=None, d_samp=None):
    if d_samp is None:
        d_samp, stats, rs_bins = gt.rejection_sample(data, proposal, sample, nbins, original_df=False)

    if plotbins == "rs":
        plotbins = rs_bins

    set_a = data.assign(set="observed")
    set_b = d_samp.assign(set="rejection sampled")
    ab = pd.concat([set_a.loc[:, [proposal, "set"]], set_b.loc[:, [proposal, "set"]]])

    if isinstance(extra_col, str):
        set_n = data.assign(set=extra_lab)
        set_n.loc[:, proposal] = set_n.loc[:, extra_col]
        ab = pd.concat([ab, set_n])

    plot = sns.histplot(ab, x=proposal, hue="set", stat="density", common_norm=False, element="step", bins=plotbins)
    return plot, d_samp

# first, some plots of different canopy metrics to assess which would be best for bias correction


def proposal_comparison_histplot(data, proposal, sample, bins="auto"):
    prop = data.loc[:, [sample]]
    prop = prop.assign(tag="all")
    samp = data.loc[~np.isnan(data.loc[:, proposal]), [sample]]
    samp = samp.assign(tag="sampled")
    ps = pd.concat([prop, samp])

    if bins == "quantile":
        scrap, bins = pd.qcut(data.loc[~np.isnan(data.loc[:, sample]), sample], q=50, retbins=True, duplicates='drop')

    plot = sns.histplot(ps, x=sample, bins=bins, hue="tag", stat="density", common_norm=False, element="step", hue_order=["sampled", "all"])
    return plot


# sample bias plots

# uf
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Ground sample bias in LPM-L distribution for 14 Feb. 2019\n Forest plot, 5cm resolution')
ax1.set_xlabel("LPM-L [-]")
ax1.set_ylabel("Relative frequency [-]")
g = proposal_comparison_histplot(data_uf, 'swe_fcon_19_045', 'lpml15', bins=30)
g.legend_.set_title(None)
legend = g.get_legend()
handles = legend.legendHandles
legend.remove()
g.legend(handles, ["limited interpolation", "all pixels"], loc="upper center")
fig.savefig(plot_out_dir + "sample_bias_lpml15_with_swe_19_045_uf.png")

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Ground sample bias in LPM-L distribution for 19 Feb. 2019\n Forest plot, 5cm resolution')
ax1.set_xlabel("LPM-L [-]")
ax1.set_ylabel("Relative frequency [-]")
g = proposal_comparison_histplot(data_uf, 'swe_fcon_19_050', 'lpml15', bins=30)
g.legend_.set_title(None)
legend = g.get_legend()
handles = legend.legendHandles
legend.remove()
g.legend(handles, ["limited interpolation", "all pixels"], loc="upper center")
fig.savefig(plot_out_dir + "sample_bias_lpml15_with_swe_19_050_uf.png")

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Ground sample bias in LPM-L distribution for 21 Feb. 2019\n Forest plot, 5cm resolution')
ax1.set_xlabel("LPM-L [-]")
ax1.set_ylabel("Relative frequency [-]")
g = proposal_comparison_histplot(data_uf, 'swe_fcon_19_052', 'lpml15', bins=30)
g.legend_.set_title(None)
legend = g.get_legend()
handles = legend.legendHandles
legend.remove()
g.legend(handles, ["limited interpolation", "all pixels"], loc="upper center")
fig.savefig(plot_out_dir + "sample_bias_lpml15_with_swe_19_052_uf.png")

# uc
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Ground sample bias in LPM-L distribution for 14 Feb. 2019\n Clearing plot, 5cm resolution')
ax1.set_xlabel("LPM-L [-]")
ax1.set_ylabel("Relative frequency [-]")
g = proposal_comparison_histplot(data_uc, 'swe_clin_19_045', 'lpml15', bins=30)
g.legend_.set_title(None)
legend = g.get_legend()
handles = legend.legendHandles
legend.remove()
g.legend(handles, ["limited interpolation", "all pixels"], loc="upper center")
fig.savefig(plot_out_dir + "sample_bias_lpml15_with_swe_19_045_uc.png")

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Ground sample bias in LPM-L distribution for 19 Feb. 2019\n Clearing plot, 5cm resolution')
ax1.set_xlabel("LPM-L [-]")
ax1.set_ylabel("Relative frequency [-]")
g = proposal_comparison_histplot(data_uc, 'swe_clin_19_050', 'lpml15', bins=30)
g.legend_.set_title(None)
legend = g.get_legend()
handles = legend.legendHandles
legend.remove()
g.legend(handles, ["limited interpolation", "all pixels"], loc="upper center")
fig.savefig(plot_out_dir + "sample_bias_lpml15_with_swe_19_050_uc.png")

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Ground sample bias in LPM-L distribution for 21 Feb. 2019\n Clearing plot, 5cm resolution')
ax1.set_xlabel("LPM-L [-]")
ax1.set_ylabel("Relative frequency [-]")
g = proposal_comparison_histplot(data_uc, 'swe_clin_19_052', 'lpml15', bins=30)
g.legend_.set_title(None)
legend = g.get_legend()
handles = legend.legendHandles
legend.remove()
g.legend(handles, ["limited interpolation", "all pixels"], loc="upper center")
fig.savefig(plot_out_dir + "sample_bias_lpml15_with_swe_19_052_uc.png")



### uf ###

#### rejection sample hs

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of snow depth (HS) for 14 Feb. 2019\n Forest plot, 5cm resolution')
ax1.set_xlabel("HS [m]")
ax1.set_ylabel("Relative frequency [-]")
g, hs_045_uf = resampling_histoplot(data_uf, '19_045_hs', 'lpml15', 50, extra_col="19_045_hs_0", extra_lab="interpolated", plotbins=200)
g.legend_.set_title(None)
legend = g.get_legend()
handles = legend.legendHandles
legend.remove()
g.legend([handles[0], handles[2], handles[1]], ["limited interpolation", "full interpolation", "rejection sampling"], loc="upper right")
fig.savefig(plot_out_dir + "freq_dist_resampled_hs_045_uf_lpml15.png")

np.nanmean(data_uf.loc[:, "19_045_hs"]) / np.nanmean(hs_045_uf.loc[:, "19_045_hs"])
np.nanmean(data_uf.loc[:, "19_045_hs_0"]) / np.nanmean(hs_045_uf.loc[:, "19_045_hs"])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of snow depth (HS) for 19 Feb. 2019\n Forest plot, 5cm resolution')
ax1.set_xlabel("HS [m]")
ax1.set_ylabel("Relative frequency [-]")
g, hs_050_uf = resampling_histoplot(data_uf, '19_050_hs', 'lpml15', 50, extra_col="19_050_hs_0", extra_lab="interpolated", plotbins=195)
g.legend_.set_title(None)
legend = g.get_legend()
handles = legend.legendHandles
legend.remove()
g.legend([handles[0], handles[2], handles[1]], ["limited interpolation", "full interpolation", "rejection sampling"], loc="upper right")
fig.savefig(plot_out_dir + "freq_dist_resampled_hs_050_uf_lpml15.png")

np.nanmean(data_uf.loc[:, "19_050_hs"]) / np.nanmean(hs_050_uf.loc[:, "19_050_hs"])
np.nanmean(data_uf.loc[:, "19_050_hs_0"]) / np.nanmean(hs_050_uf.loc[:, "19_050_hs"])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of snow depth (HS) for 21 Feb. 2019\n Forest plot, 5cm resolution')
ax1.set_xlabel("HS [m]")
ax1.set_ylabel("Relative frequency [-]")
g, hs_052_uf = resampling_histoplot(data_uf, '19_052_hs', 'lpml15', 50, extra_col="19_052_hs_0", extra_lab="interpolated", plotbins=200)
g.legend_.set_title(None)
legend = g.get_legend()
handles = legend.legendHandles
legend.remove()
g.legend([handles[0], handles[2], handles[1]], ["limited interpolation", "full interpolation", "rejection sampling"], loc="upper right")
fig.savefig(plot_out_dir + "freq_dist_resampled_hs_052_uf_lpml15.png")

np.nanmean(data_uf.loc[:, "19_052_hs"]) / np.nanmean(hs_052_uf.loc[:, "19_052_hs"])
np.nanmean(data_uf.loc[:, "19_052_hs_0"]) / np.nanmean(hs_052_uf.loc[:, "19_052_hs"])


# plot all together
hs_045_uf = hs_045_uf.assign(date="14 Feb", hs=data_uf.loc[:, "19_045_hs"])
hs_050_uf = hs_050_uf.assign(date="19 Feb", hs=data_uf.loc[:, "19_050_hs"])
hs_052_uf = hs_052_uf.assign(date="21 Feb", hs=data_uf.loc[:, "19_052_hs"])
all_hs_uf = pd.concat([hs_045_uf.loc[:, ["hs", "date"]], hs_050_uf.loc[:, ["hs", "date"]], hs_052_uf.loc[:, ["hs", "date"]]])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distributions of snow depth (HS) for all days\n Forest plot, 5cm resolution, rejection sampled with LPM-L')
ax1.set_xlabel("HS [m]")
ax1.set_ylabel("Relative frequency [-]")
plot = sns.histplot(all_hs_uf, x="hs", hue="date", stat="density", common_norm=False, element="step", bins=196)
fig.savefig(plot_out_dir + "freq_dist_resampled_all_hs_uf_lpml15.png")

# export rejection sampled points
e_hs_045_uf = hs_045_uf.loc[:, ["x_coord", "y_coord", "19_045_hs"]]
e_hs_050_uf = hs_050_uf.loc[:, ["x_coord", "y_coord", "19_050_hs"]]
e_hs_052_uf = hs_052_uf.loc[:, ["x_coord", "y_coord", "19_052_hs"]]

e_hs_045_uf.to_csv(rej_samp_out_file + 'resampled_hs_19_045_uf_r.10m_interp2x_by_lpml15.csv', index=False)
e_hs_050_uf.to_csv(rej_samp_out_file + 'resampled_hs_19_050_uf_r.10m_interp2x_by_lpml15.csv', index=False)
e_hs_052_uf.to_csv(rej_samp_out_file + 'resampled_hs_19_052_uf_r.10m_interp2x_by_lpml15.csv', index=False)
#
# # # # # Reload samples
# hs_045_uf = pd.read_csv(rej_samp_out_file + 'resampled_hs_19_045_uf_r.10m_interp2x_by_lpml15.csv')
# hs_050_uf = pd.read_csv(rej_samp_out_file + 'resampled_hs_19_050_uf_r.10m_interp2x_by_lpml15.csv')
# hs_052_uf = pd.read_csv(rej_samp_out_file + 'resampled_hs_19_052_uf_r.10m_interp2x_by_lpml15.csv')
# stats

# acceptance rates
np.sum(~np.isnan(hs_045_uf.loc[:, "19_045_hs"])) / np.sum(~np.isnan(data_uf.loc[:, "19_045_hs"]))
np.sum(~np.isnan(hs_050_uf.loc[:, "19_050_hs"])) / np.sum(~np.isnan(data_uf.loc[:, "19_050_hs"]))
np.sum(~np.isnan(hs_052_uf.loc[:, "19_052_hs"])) / np.sum(~np.isnan(data_uf.loc[:, "19_052_hs"]))


# limited interpolation coverage
np.sum(~np.isnan(data_uf.loc[:, "19_045_hs"])) / np.sum(~np.isnan(data_uf.loc[:, "19_045_hs_0"]))
np.sum(~np.isnan(data_uf.loc[:, "19_050_hs"])) / np.sum(~np.isnan(data_uf.loc[:, "19_050_hs_0"]))
np.sum(~np.isnan(data_uf.loc[:, "19_052_hs"])) / np.sum(~np.isnan(data_uf.loc[:, "19_052_hs_0"]))

#
# mean
np.nanmean(data_uf.loc[:, "19_045_hs"]) / np.nanmean(hs_045_uf.loc[:, "19_045_hs"])
np.nanmean(data_uf.loc[:, "19_045_hs_0"]) / np.nanmean(hs_045_uf.loc[:, "19_045_hs"])

np.nanmean(data_uf.loc[:, "19_050_hs"]) / np.nanmean(hs_050_uf.loc[:, "19_050_hs"])
np.nanmean(data_uf.loc[:, "19_050_hs_0"]) / np.nanmean(hs_050_uf.loc[:, "19_050_hs"])

np.nanmean(data_uf.loc[:, "19_052_hs"]) / np.nanmean(hs_052_uf.loc[:, "19_052_hs"])
np.nanmean(data_uf.loc[:, "19_052_hs_0"]) / np.nanmean(hs_052_uf.loc[:, "19_052_hs"])

# var
np.nanvar(data_uf.loc[:, "19_045_hs"]) / np.nanvar(hs_045_uf.loc[:, "19_045_hs"])
np.nanvar(data_uf.loc[:, "19_045_hs_0"]) / np.nanvar(hs_045_uf.loc[:, "19_045_hs"])

np.nanvar(data_uf.loc[:, "19_050_hs"]) / np.nanvar(hs_050_uf.loc[:, "19_050_hs"])
np.nanvar(data_uf.loc[:, "19_050_hs_0"]) / np.nanvar(hs_050_uf.loc[:, "19_050_hs"])

np.nanvar(data_uf.loc[:, "19_052_hs"]) / np.nanvar(hs_052_uf.loc[:, "19_052_hs"])
np.nanvar(data_uf.loc[:, "19_052_hs_0"]) / np.nanvar(hs_052_uf.loc[:, "19_052_hs"])

# std
np.nanstd(data_uf.loc[:, "19_045_hs"]) / np.nanstd(hs_045_uf.loc[:, "19_045_hs"])
np.nanstd(data_uf.loc[:, "19_045_hs_0"]) / np.nanstd(hs_045_uf.loc[:, "19_045_hs"])

np.nanstd(data_uf.loc[:, "19_050_hs"]) / np.nanstd(hs_050_uf.loc[:, "19_050_hs"])
np.nanstd(data_uf.loc[:, "19_050_hs_0"]) / np.nanstd(hs_050_uf.loc[:, "19_050_hs"])

np.nanstd(data_uf.loc[:, "19_052_hs"]) / np.nanstd(hs_052_uf.loc[:, "19_052_hs"])
np.nanstd(data_uf.loc[:, "19_052_hs_0"]) / np.nanstd(hs_052_uf.loc[:, "19_052_hs"])


#
# from scipy import stats
# # a = hs_045_uf.loc[:, "19_045_hs"]
# # a = hs_050_uf.loc[:, "19_050_hs"]
# a = hs_052_uf.loc[:, "19_052_hs"]
# a = a[~np.isnan(a)]
# # a = a[a > 0]
# # a = np.log(a)
# # b = data_uf.loc[:, "19_045_hs_0"]
# # b = data_uf.loc[:, "19_050_hs_0"]
# # b = data_uf.loc[:, "19_052_hs_0"]
# # b = data_uf.loc[:, "19_045_hs"]
# # b = data_uf.loc[:, "19_050_hs"]
# b = data_uf.loc[:, "19_052_hs"]
# # b = b[b > 0]
# # n = np.log(b)
# b = b[~np.isnan(b)]
#
#
# tStat, pValue = stats.levene(a, b, center='median') #run independent sample T-Test
# print("Levene: P-Value:{0} T-Statistic:{1}".format(pValue, tStat)) #print the P-Value and the T-Statistic
# tStat, pValue = stats.ttest_ind(a, b, equal_var=False) #run independent sample T-Test
# print("T-test: P-Value:{0} T-Statistic:{1}".format(pValue, tStat)) #print the P-Value and the T-Statistic
#

#### rejection sample swe
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of SWE for 14 Feb. 2019\n Forest plot, 5cm resolution, rejection sampled with LPM-L')
ax1.set_xlabel("SWE [mm]")
ax1.set_ylabel("Relative frequency [-]")
g, d_045_uf = resampling_histoplot(data_uf, 'swe_fcon_19_045', 'lpml15', 50)
g.legend_.set_title(None)
fig.savefig(plot_out_dir + "freq_dist_resampled_swe_045_uf_lpml15.png")

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of SWE for 19 Feb. 2019\n Forest plot, 5cm resolution, rejection sampled with LPM-L')
ax1.set_xlabel("SWE [mm]")
ax1.set_ylabel("Relative frequency [-]")
g, d_050_uf = resampling_histoplot(data_uf, 'swe_fcon_19_050', 'lpml15', 50)
g.legend_.set_title(None)
fig.savefig(plot_out_dir + "freq_dist_resampled_swe_050_uf_lpml15.png")

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of SWE for 21 Feb. 2019\n Forest plot, 5cm resolution, rejection sampled with LPM-L')
ax1.set_xlabel("SWE [mm]")
ax1.set_ylabel("Relative frequency [-]")
g, d_052_uf = resampling_histoplot(data_uf, 'swe_fcon_19_052', 'lpml15', 50)
g.legend_.set_title(None)
fig.savefig(plot_out_dir + "freq_dist_resampled_swe_052_uf_lpml15.png")

# plot all together
d_045_uf = d_045_uf.assign(date="14 Feb", swe=d_045_uf.swe_fcon_19_045)
d_050_uf = d_050_uf.assign(date="19 Feb", swe=d_050_uf.swe_fcon_19_050)
d_052_uf = d_052_uf.assign(date="21 Feb", swe=d_052_uf.swe_fcon_19_052)
all_swe_uf = pd.concat([d_045_uf.loc[:, ["swe", "date"]], d_050_uf.loc[:, ["swe", "date"]], d_052_uf.loc[:, ["swe", "date"]]])
# all_swe_uf = pd.concat([d_050_uf.loc[:, ["swe", "date"]], d_052_uf.loc[:, ["swe", "date"]]])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distributions of SWE for all days\n Forest plot, 5cm resolution, rejection sampled with LPM-L')
ax1.set_xlabel("SWE [mm]")
ax1.set_ylabel("Relative frequency [-]")
plot = sns.histplot(all_swe_uf, x="swe", hue="date", stat="density", common_norm=False, element="step")
fig.savefig(plot_out_dir + "freq_dist_resampled_all_swe_uf_lpml15.png")

# export rejection sampled points
e_045_uf = d_045_uf.loc[:, ["x_coord", "y_coord", "swe_fcon_19_045"]]
e_050_uf = d_050_uf.loc[:, ["x_coord", "y_coord", "swe_fcon_19_050"]]
e_052_uf = d_052_uf.loc[:, ["x_coord", "y_coord", "swe_fcon_19_052"]]

e_045_uf.to_csv(rej_samp_out_file + 'resampled_swe_19_045_uf_fcon_r.05m_interp2x_by_lpml15.csv', index=False)
e_050_uf.to_csv(rej_samp_out_file + 'resampled_swe_19_050_uf_fcon_r.05m_interp2x_by_lpml15.csv', index=False)
e_052_uf.to_csv(rej_samp_out_file + 'resampled_swe_19_052_uf_fcon_r.05m_interp2x_by_lpml15.csv', index=False)

# reload rejection samples
swe_045_uf = pd.read_csv(rej_samp_out_file + 'resampled_swe_19_045_uf_fcon_r.05m_interp2x_by_lpml15.csv')
swe_050_uf = pd.read_csv(rej_samp_out_file + 'resampled_swe_19_050_uf_fcon_r.05m_interp2x_by_lpml15.csv')
swe_052_uf = pd.read_csv(rej_samp_out_file + 'resampled_swe_19_052_uf_fcon_r.05m_interp2x_by_lpml15.csv')

#### rejection sample dswe
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of $\Delta$SWE for 14-19 Feb. 2019\n Forest plot, 5cm resolution, rejection sampled with LPM-L')
ax1.set_xlabel("SWE [mm]")
ax1.set_ylabel("Relative frequency [-]")
g, d_045_050_uf = resampling_histoplot(data_uf, 'dswe_fnsd_19_045-19_050', 'lpml15', 50, plotbins=250)
g.legend_.set_title(None)
fig.savefig(plot_out_dir + "freq_dist_resampled_dswe_045-050_uf_lpml15.png")

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of $\Delta$SWE for 19-21 Feb. 2019\n Forest plot, 5cm resolution, rejection sampled with LPM-L')
ax1.set_xlabel("SWE [mm]")
ax1.set_ylabel("Relative frequency [-]")
g, d_050_052_uf = resampling_histoplot(data_uf, 'dswe_fnsd_19_050-19_052', 'lpml15', 50, plotbins=200)
g.legend_.set_title(None)
fig.savefig(plot_out_dir + "freq_dist_resampled_dswe_050-052_uf_lpml15.png")

# plot all together
d_045_050_uf = d_045_050_uf.assign(interval="14-19 Feb", dswe=d_045_050_uf.loc[:, "dswe_fnsd_19_045-19_050"])
d_050_052_uf = d_050_052_uf.assign(interval="19-21 Feb", dswe=d_050_052_uf.loc[:, "dswe_fnsd_19_050-19_052"])
all_dswe_uf = pd.concat([d_045_050_uf.loc[:, ["dswe", "interval"]], d_050_052_uf.loc[:, ["dswe", "interval"]]])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distributions of $\Delta$SWE for all days\n Forest plot, 5cm resolution, rejection sampled with LPM-L')
ax1.set_xlabel("$\Delta$SWE [mm]")
ax1.set_ylabel("Relative frequency [-]")
sns.histplot(all_dswe_uf, x="dswe", hue="interval", stat="density", common_norm=False, element="step", bins=200)
fig.savefig(plot_out_dir + "freq_dist_resampled_all_dswe_uf_lpml15.png")

# export rejection sampled points
e_045_050_uf = d_045_050_uf.loc[:, ["x_coord", "y_coord", "dswe_fnsd_19_045-19_050"]]
e_050_052_uf = d_050_052_uf.loc[:, ["x_coord", "y_coord", "dswe_fnsd_19_050-19_052"]]

e_045_050_uf.to_csv(rej_samp_out_file + 'resampled_dswe_19_045-19_050_uf_fnsd_r.05m_interp2x_by_lpml15.csv', index=False)
e_050_052_uf.to_csv(rej_samp_out_file + 'resampled_dswe_19_050-19_052_uf_fnsd_r.05m_interp2x_by_lpml15.csv', index=False)


### uc ###

#### rejection sample hs
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of snow depth (HS) for 14 Feb. 2019\n Clearing plot, 5cm resolution')
ax1.set_xlabel("HS [m]")
ax1.set_ylabel("Relative frequency [-]")
g, hs_045_uc = resampling_histoplot(data_uc, '19_045_hs', 'lpml15', 50, extra_col="19_045_hs_0", extra_lab="interpolated", plotbins=195)
g.legend_.set_title(None)
legend = g.get_legend()
handles = legend.legendHandles
legend.remove()
g.legend([handles[0], handles[2], handles[1]], ["limited interpolation", "full interpolation", "rejection sampling"], loc="upper left")
fig.savefig(plot_out_dir + "freq_dist_resampled_hs_045_uc_lpml15.png")


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of snow depth (HS) for 19 Feb. 2019\n Clearing plot, 5cm resolution')
ax1.set_xlabel("HS [m]")
ax1.set_ylabel("Relative frequency [-]")
g, hs_050_uc = resampling_histoplot(data_uc, '19_050_hs', 'lpml15', 50, extra_col="19_050_hs_0", extra_lab="interpolated", plotbins=194)
g.legend_.set_title(None)
legend = g.get_legend()
handles = legend.legendHandles
legend.remove()
g.legend([handles[0], handles[2], handles[1]], ["limited interpolation", "full interpolation", "rejection sampling"], loc="upper left")
fig.savefig(plot_out_dir + "freq_dist_resampled_hs_050_uc_lpml15.png")


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of snow depth (HS) for 21 Feb. 2019\n Clearing plot, 5cm resolution')
ax1.set_xlabel("HS [m]")
ax1.set_ylabel("Relative frequency [-]")
g, hs_052_uc = resampling_histoplot(data_uc, '19_052_hs', 'lpml15', 50, extra_col="19_052_hs_0", extra_lab="interpolated", plotbins=199)
g.legend_.set_title(None)
legend = g.get_legend()
handles = legend.legendHandles
legend.remove()
g.legend([handles[0], handles[2], handles[1]], ["limited interpolation", "full interpolation", "rejection sampling"], loc="upper left")
fig.savefig(plot_out_dir + "freq_dist_resampled_hs_052_uc_lpml15.png")


# plot all together
hs_045_uc = hs_045_uc.assign(date="14 Feb", hs=data_uc.loc[:, "19_045_hs"])
hs_050_uc = hs_050_uc.assign(date="19 Feb", hs=data_uc.loc[:, "19_050_hs"])
hs_052_uc = hs_052_uc.assign(date="21 Feb", hs=data_uc.loc[:, "19_052_hs"])
all_hs_uc = pd.concat([hs_045_uc.loc[:, ["hs", "date"]], hs_050_uc.loc[:, ["hs", "date"]], hs_052_uc.loc[:, ["hs", "date"]]])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distributions of snow depth (HS) for all days\n Clearing plot, 5cm resolution, rejection sampled with LPM-L')
ax1.set_xlabel("HS [m]")
ax1.set_ylabel("Relative frequency [-]")
plot = sns.histplot(all_hs_uc, x="hs", hue="date", stat="density", common_norm=False, element="step", bins=202)
fig.savefig(plot_out_dir + "freq_dist_resampled_all_hs_uc_lpml15.png")

# export rejection sampled points
e_hs_045_uc = hs_045_uc.loc[:, ["x_coord", "y_coord", "19_045_hs"]]
e_hs_050_uc = hs_050_uc.loc[:, ["x_coord", "y_coord", "19_050_hs"]]
e_hs_052_uc = hs_052_uc.loc[:, ["x_coord", "y_coord", "19_052_hs"]]

e_hs_045_uc.to_csv(rej_samp_out_file + 'resampled_hs_19_045_uc_r.10m_interp2x_by_lpml15.csv', index=False)
e_hs_050_uc.to_csv(rej_samp_out_file + 'resampled_hs_19_050_uc_r.10m_interp2x_by_lpml15.csv', index=False)
e_hs_052_uc.to_csv(rej_samp_out_file + 'resampled_hs_19_052_uc_r.10m_interp2x_by_lpml15.csv', index=False)


# # # Reload samples
hs_045_uc = pd.read_csv(rej_samp_out_file + 'resampled_hs_19_045_uc_r.10m_interp2x_by_lpml15.csv')
hs_050_uc = pd.read_csv(rej_samp_out_file + 'resampled_hs_19_050_uc_r.10m_interp2x_by_lpml15.csv')
hs_052_uc = pd.read_csv(rej_samp_out_file + 'resampled_hs_19_052_uc_r.10m_interp2x_by_lpml15.csv')

# # stats
#

# acceptance rates
np.sum(~np.isnan(hs_045_uc.loc[:, "19_045_hs"])) / np.sum(~np.isnan(data_uc.loc[:, "19_045_hs"]))
np.sum(~np.isnan(hs_050_uc.loc[:, "19_050_hs"])) / np.sum(~np.isnan(data_uc.loc[:, "19_050_hs"]))
np.sum(~np.isnan(hs_052_uc.loc[:, "19_052_hs"])) / np.sum(~np.isnan(data_uc.loc[:, "19_052_hs"]))


# limited interpolation coverage
np.sum(~np.isnan(data_uc.loc[:, "19_045_hs"])) / np.sum(~np.isnan(data_uc.loc[:, "19_045_hs_0"]))
np.sum(~np.isnan(data_uc.loc[:, "19_050_hs"])) / np.sum(~np.isnan(data_uc.loc[:, "19_050_hs_0"]))
np.sum(~np.isnan(data_uc.loc[:, "19_052_hs"])) / np.sum(~np.isnan(data_uc.loc[:, "19_052_hs_0"]))
#
# mean
np.nanmean(data_uc.loc[:, "19_045_hs"]) / np.nanmean(hs_045_uc.loc[:, "19_045_hs"])
np.nanmean(data_uc.loc[:, "19_045_hs_0"]) / np.nanmean(hs_045_uc.loc[:, "19_045_hs"])

np.nanmean(data_uc.loc[:, "19_050_hs"]) / np.nanmean(hs_050_uc.loc[:, "19_050_hs"])
np.nanmean(data_uc.loc[:, "19_050_hs_0"]) / np.nanmean(hs_050_uc.loc[:, "19_050_hs"])

np.nanmean(data_uc.loc[:, "19_052_hs"]) / np.nanmean(hs_052_uc.loc[:, "19_052_hs"])
np.nanmean(data_uc.loc[:, "19_052_hs_0"]) / np.nanmean(hs_052_uc.loc[:, "19_052_hs"])

# var
np.nanvar(data_uc.loc[:, "19_045_hs"]) / np.nanvar(hs_045_uc.loc[:, "19_045_hs"])
np.nanvar(data_uc.loc[:, "19_045_hs_0"]) / np.nanvar(hs_045_uc.loc[:, "19_045_hs"])

np.nanvar(data_uc.loc[:, "19_050_hs"]) / np.nanvar(hs_050_uc.loc[:, "19_050_hs"])
np.nanvar(data_uc.loc[:, "19_050_hs_0"]) / np.nanvar(hs_050_uc.loc[:, "19_050_hs"])

np.nanvar(data_uc.loc[:, "19_052_hs"]) / np.nanvar(hs_052_uc.loc[:, "19_052_hs"])
np.nanvar(data_uc.loc[:, "19_052_hs_0"]) / np.nanvar(hs_052_uc.loc[:, "19_052_hs"])

# std
np.nanstd(data_uc.loc[:, "19_045_hs"]) / np.nanstd(hs_045_uc.loc[:, "19_045_hs"])
np.nanstd(data_uc.loc[:, "19_045_hs_0"]) / np.nanstd(hs_045_uc.loc[:, "19_045_hs"])

np.nanstd(data_uc.loc[:, "19_050_hs"]) / np.nanstd(hs_050_uc.loc[:, "19_050_hs"])
np.nanstd(data_uc.loc[:, "19_050_hs_0"]) / np.nanstd(hs_050_uc.loc[:, "19_050_hs"])

np.nanstd(data_uc.loc[:, "19_052_hs"]) / np.nanstd(hs_052_uc.loc[:, "19_052_hs"])
np.nanstd(data_uc.loc[:, "19_052_hs_0"]) / np.nanstd(hs_052_uc.loc[:, "19_052_hs"])
#
# from scipy import stats
# # a = hs_045_uc.loc[:, "19_045_hs"]
# # a = hs_050_uc.loc[:, "19_050_hs"]
# a = hs_052_uc.loc[:, "19_052_hs"]
# a = a[~np.isnan(a)]
# # b = data_uc.loc[:, "19_045_hs_0"]
# # b = data_uc.loc[:, "19_050_hs_0"]
# # b = data_uc.loc[:, "19_052_hs_0"]
# # b = data_uc.loc[:, "19_045_hs"]
# # b = data_uc.loc[:, "19_050_hs"]
# b = data_uc.loc[:, "19_052_hs"]
# b = b[~np.isnan(b)]
#
# tStat, pValue = stats.levene(a, b, center='median') #run independent sample T-Test
# print("Levene: P-Value:{0} T-Statistic:{1}".format(pValue, tStat)) #print the P-Value and the T-Statistic
# tStat, pValue = stats.ttest_ind(a, b, equal_var=False) #run independent sample T-Test
# print("T-test: P-Value:{0} T-Statistic:{1}".format(pValue, tStat)) #print the P-Value and the T-Statistic


#### rejection sample swe
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of SWE for 14 Feb. 2019\n Clearing plot, 5cm resolution, rejection sampled with LPM-L')
ax1.set_xlabel("SWE [mm]")
ax1.set_ylabel("Relative frequency [-]")
g, d_045_uc = resampling_histoplot(data_uc, 'swe_clin_19_045', 'lpml15', 50)
g.legend_.set_title(None)
fig.savefig(plot_out_dir + "freq_dist_resampled_swe_045_uc_lpml15.png")

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of SWE for 19 Feb. 2019\n Clearing plot, 5cm resolution, rejection sampled with LPM-L')
ax1.set_xlabel("SWE [mm]")
ax1.set_ylabel("Relative frequency [-]")
g, d_050_uc = resampling_histoplot(data_uc, 'swe_clin_19_050', 'lpml15', 50)
g.legend_.set_title(None)
fig.savefig(plot_out_dir + "freq_dist_resampled_swe_050_uc_lpml15.png")

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of SWE for 21 Feb. 2019\n Clearing plot, 5cm resolution, rejection sampled with LPM-L')
ax1.set_xlabel("SWE [mm]")
ax1.set_ylabel("Relative frequency [-]")
g, d_052_uc = resampling_histoplot(data_uc, 'swe_clin_19_052', 'lpml15', 50)
g.legend_.set_title(None)
fig.savefig(plot_out_dir + "freq_dist_resampled_swe_052_uc_lpml15.png")

# plot all together
d_045_uc = d_045_uc.assign(date="14 Feb", swe=d_045_uc.swe_clin_19_045)
d_050_uc = d_050_uc.assign(date="19 Feb", swe=d_050_uc.swe_clin_19_050)
d_052_uc = d_052_uc.assign(date="21 Feb", swe=d_052_uc.swe_clin_19_052)
all_swe_uc = pd.concat([d_045_uc.loc[:, ["swe", "date"]], d_050_uc.loc[:, ["swe", "date"]], d_052_uc.loc[:, ["swe", "date"]]])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distributions of SWE for all days\n Clearing plot, 5cm resolution, rejection sampled with LPM-L')
ax1.set_xlabel("SWE [mm]")
ax1.set_ylabel("Relative frequency [-]")
plot = sns.histplot(all_swe_uc, x="swe", hue="date", stat="density", common_norm=False, element="step")
fig.savefig(plot_out_dir + "freq_dist_resampled_all_swe_uc_lpml15.png")

# export rejection sampled points
e_045_uc = d_045_uc.loc[:, ["x_coord", "y_coord", "swe_clin_19_045"]]
e_050_uc = d_050_uc.loc[:, ["x_coord", "y_coord", "swe_clin_19_050"]]
e_052_uc = d_052_uc.loc[:, ["x_coord", "y_coord", "swe_clin_19_052"]]

e_045_uc.to_csv(rej_samp_out_file + 'resampled_swe_19_045_uc_clin_r.05m_interp2x_by_lpml15.csv', index=False)
e_050_uc.to_csv(rej_samp_out_file + 'resampled_swe_19_050_uc_clin_r.05m_interp2x_by_lpml15.csv', index=False)
e_052_uc.to_csv(rej_samp_out_file + 'resampled_swe_19_052_uc_clin_r.05m_interp2x_by_lpml15.csv', index=False)


#### rejection sample dswe
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of $\Delta$SWE for 14-19 Feb. 2019\n Clearing plot, 5cm resolution, rejection sampled with LPM-L')
ax1.set_xlabel("SWE [mm]")
ax1.set_ylabel("Relative frequency [-]")
g, d_045_050_uc = resampling_histoplot(data_uc, 'dswe_cnsd_19_045-19_050', 'lpml15', 50, plotbins=250)
g.legend_.set_title(None)
fig.savefig(plot_out_dir + "freq_dist_resampled_dswe_045-050_uc_lpml15.png")

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of $\Delta$SWE for 19-21 Feb. 2019\n Clearing plot, 5cm resolution, rejection sampled with LPM-L')
ax1.set_xlabel("SWE [mm]")
ax1.set_ylabel("Relative frequency [-]")
g, d_050_052_uc = resampling_histoplot(data_uc, 'dswe_cnsd_19_050-19_052', 'lpml15', 50, plotbins=250)
g.legend_.set_title(None)
fig.savefig(plot_out_dir + "freq_dist_resampled_dswe_050-052_uc_lpml15.png")

# plot all together
d_045_050_uc = d_045_050_uc.assign(interval="14-19 Feb", dswe=d_045_050_uc.loc[:, "dswe_cnsd_19_045-19_050"])
d_050_052_uc = d_050_052_uc.assign(interval="19-21 Feb", dswe=d_050_052_uc.loc[:, "dswe_cnsd_19_050-19_052"])
all_dswe_uc = pd.concat([d_045_050_uc.loc[:, ["dswe", "interval"]], d_050_052_uc.loc[:, ["dswe", "interval"]]])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distributions of $\Delta$SWE for all days\n Clearing plot, 5cm resolution, rejection sampled with LPM-L')
ax1.set_xlabel("$\Delta$SWE [mm]")
ax1.set_ylabel("Relative frequency [-]")
sns.histplot(all_dswe_uc, x="dswe", hue="interval", stat="density", common_norm=False, element="step", bins=250)
fig.savefig(plot_out_dir + "freq_dist_resampled_all_dswe_uc_lpml15.png")

# export rejection sampled points
e_045_050_uc = d_045_050_uc.loc[:, ["x_coord", "y_coord", "dswe_cnsd_19_045-19_050"]]
e_050_052_uc = d_050_052_uc.loc[:, ["x_coord", "y_coord", "dswe_cnsd_19_050-19_052"]]

e_045_050_uc.to_csv(rej_samp_out_file + 'resampled_dswe_19_045-19_050_uc_cnsd_r.05m_interp2x_by_lpml15.csv', index=False)
e_050_052_uc.to_csv(rej_samp_out_file + 'resampled_dswe_19_050-19_052_uc_cnsd_r.05m_interp2x_by_lpml15.csv', index=False)

# combined plot plots
# plot 045-050 all together
d_045_050_uf = d_045_050_uf.assign(plot="upper forest", dswe=d_045_050_uf.loc[:, "dswe_fnsd_19_045-19_050"])
d_045_050_uc = d_045_050_uc.assign(plot="upper clearing", dswe=d_045_050_uc.loc[:, "dswe_cnsd_19_045-19_050"])
d_045_050_dswe = pd.concat([d_045_050_uf.loc[:, ["dswe", "plot"]], d_045_050_uc.loc[:, ["dswe", "plot"]]])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distributions of $\Delta$SWE for 14-19 Feb. 2019\n 5cm resolution, rejection sampled with LPM-L')
ax1.set_xlabel("$\Delta$SWE [mm]")
ax1.set_ylabel("Relative frequency [-]")
sns.histplot(d_045_050_dswe, x="dswe", hue="plot", stat="density", common_norm=False, element="step", bins=300)
fig.savefig(plot_out_dir + "freq_dist_resampled_045_050_dswe_lpml15.png")

# plot 050-052 all together
d_050_052_uf = d_050_052_uf.assign(plot="upper forest", dswe=d_050_052_uf.loc[:, "dswe_fnsd_19_050-19_052"])
d_050_052_uc = d_050_052_uc.assign(plot="upper clearing", dswe=d_050_052_uc.loc[:, "dswe_cnsd_19_050-19_052"])
d_050_052_dswe = pd.concat([d_050_052_uf.loc[:, ["dswe", "plot"]], d_050_052_uc.loc[:, ["dswe", "plot"]]])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distributions of $\Delta$SWE for 19-21 Feb. 2019\n 5cm resolution, rejection sampled with LPM-L')
ax1.set_xlabel("$\Delta$SWE [mm]")
ax1.set_ylabel("Relative frequency [-]")
sns.histplot(d_050_052_dswe, x="dswe", hue="plot", stat="density", common_norm=False, element="step", bins=300)
fig.savefig(plot_out_dir + "freq_dist_resampled_050_052_dswe_lpml15.png")

##### lai and transmittance
data_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\merged_data_products\\merged_mb_15_r.25m_canopy_19_149.csv'
c_data = pd.read_csv(data_in)
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
ax1.set_title('Frequency distributions of LAI across site\n 25cm resolution, snow-free canopy')
ax1.set_xlabel("LAI [-]")
ax1.set_ylabel("Relative frequency [-]")
sns.histplot(ab, x="lai", hue="method", stat="density", common_norm=False, element="step")
plt.xlim(0, 2)
fig.savefig(plot_out_dir + "freq_dist_lai_mb_15.png")

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
ax1.set_title('Frequency distributions of light transmittance across site\n25cm resolution, snow-free canopy')
ax1.set_xlabel("Transmittance [-]")
ax1.set_ylabel("Relative frequency [-]")
g = sns.histplot(ab, x="trans", hue="method", stat="density", common_norm=False, element="step", bins=70)
g.legend_.set_title(None)
legend = g.get_legend()
handles = legend.legendHandles
legend.remove()
g.legend(handles, ["Ray Sampling 1deg", "Hemi-photo 15deg", "Hemi-photo 75deg"], loc="upper center")
fig.savefig(plot_out_dir + "freq_dist_trans_mb_15.png")


### other canopy metrics
data_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\merged_data_products\\merged_mb_15_r.10m_canopy_19_149.csv'
cc_data = pd.read_csv(data_in)
# filter to upper forest
cc_data = cc_data.loc[cc_data.plots == 1, :]  # uf only

# DNT
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of Distance to Nearest Tree (DNT)\n Forest plot, 10cm resolution, snow-free canopy')
ax1.set_xlabel("DNT [m]")
ax1.set_ylabel("Relative frequency [-]")
sns.histplot(cc_data, x="dnt", stat="density", element="step")
fig.savefig(plot_out_dir + "freq_dist_dnt_uf.png")

# DCE
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of Distance from Canopy Edge (DCE)\n Forest plot, 10cm resolution, snow-free canopy')
ax1.set_xlabel("DCE [m]")
ax1.set_ylabel("Relative frequency [-]")
sns.histplot(cc_data, x="dce", stat="density", element="step", binwidth=0.1)
fig.savefig(plot_out_dir + "freq_dist_dce_uf.png")

# CHM
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distribution of vegetation crown height\n Forest plot, 10cm resolution, snow-free vegetation > 1m')
ax1.set_xlabel("Vegetation crown height [m]")
ax1.set_ylabel("Relative frequency [-]")
sns.histplot(cc_data.loc[cc_data.chm > 1, ], x="chm", stat="density", element="step")
fig.savefig(plot_out_dir + "freq_dist_chm_uf.png")

# mCH
set_a = cc_data.assign(method="Snow-off raw", mch=cc_data.mCH_19_149[cc_data.mCH_19_149 > 0])
set_b = cc_data.assign(method="Snow-off resampled", mch=cc_data.mCH_19_149_resampled[cc_data.mCH_19_149_resampled > 0])
set_c = cc_data.assign(method="Snow on resampled", mch=cc_data.mCH_045_050_052_resampled[cc_data.mCH_045_050_052_resampled > 0])
ab = pd.concat([set_a.loc[:, ["mch", "method"]], set_b.loc[:, ["mch", "method"]]])
bc = pd.concat([set_b.loc[:, ["mch", "method"]],  set_c.loc[:, ["mch", "method"]]])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distributions of mean canopy height (mCH)\n Forest plot, 10cm resolution')
ax1.set_xlabel("Mean canopy height [m]")
ax1.set_ylabel("Relative frequency [-]")
sns.histplot(ab, x="mch", hue="method", stat="density", common_norm=False, element="step", bins=100)
fig.savefig(plot_out_dir + "freq_dist_mch_snow_off_uf.png")

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distributions of mean canopy height (mCH)\n Forest plot, 10cm resolution')
ax1.set_xlabel("Mean canopy height [m]")
ax1.set_ylabel("Relative frequency [-]")
sns.histplot(bc, x="mch", hue="method", stat="density", common_norm=False, element="step", bins=100)
fig.savefig(plot_out_dir + "freq_dist_mch_resampled_uf.png")


# LPMs
set_a = cc_data.assign(method="First", lpm=cc_data.lpmf15)
set_b = cc_data.assign(method="Last", lpm=cc_data.lpml15)
set_c = cc_data.assign(method="Canopy", lpm=cc_data.lpmc15)
ab = pd.concat([set_a.loc[:, ["lpm", "method"]], set_b.loc[:, ["lpm", "method"]],  set_c.loc[:, ["lpm", "method"]]])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Frequency distributions of light transmittance\n Forest plot, 10cm resolution, snow-free canopy')
ax1.set_xlabel("Transmittance [-]")
ax1.set_ylabel("Relative frequency [-]")
g = sns.histplot(ab, x="lpm", hue="method", stat="density", common_norm=False, element="step", bins=40)
g.legend_.set_title(None)
legend = g.get_legend()
handles = legend.legendHandles
legend.remove()
g.legend(handles, ["LPM-First", "LPM-Last", "LPM-Canopy"], loc="upper center")
fig.savefig(plot_out_dir + "freq_dist_trans_uf.png")

# plotting samples in space

# need raster template of upper forest..
uf_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\site_library\\hemi_grid_points\\mb_65_r.25m_snow_off_offset0\\uf_plot_r.05m.tif'
uf = raslib.raster_load(uf_in)

min_p = np.min(np.where(uf.data), axis=1)
max_p = np.max(np.where(uf.data), axis=1)

full = uf.data.copy()

limited = uf.data.copy()
limited[:, :] = 0
limited[data_uf.y_index[~np.isnan(data_uf.loc[:, "19_050_hs"])], data_uf.x_index[~np.isnan(data_uf.loc[:, "19_050_hs"])]] = 1

resamp = uf.data.copy()
resamp[:, :] = 0
resamp[hs_050_uf.y_index, hs_050_uf.x_index] = 1


lpm = np.full(np.shape(uf.data), np.nan)
lpm[data_uf.y_index, data_uf.x_index] = data_uf.lpml15

fig = plt.figure()
plt.imshow(full[min_p[0]:max_p[0], min_p[1]:max_p[1]], cmap="Greens")


fig = plt.figure()
plt.imshow(limited[min_p[0]:max_p[0], min_p[1]:max_p[1]], cmap="Blues")

fig = plt.figure()
plt.imshow(resamp[min_p[0]:max_p[0], min_p[1]:max_p[1]], cmap="Oranges")

fig = plt.figure()
plt.imshow(lpm[min_p[0]:max_p[0], min_p[1]:max_p[1]], cmap="Reds")
plt.colorbar()