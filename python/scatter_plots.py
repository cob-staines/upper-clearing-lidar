import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

plot_out_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\graphics\\thesis_graphics\\scatter plots\\"

df_10_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\merged_data_products\\merged_uf_r.10m_canopy_19_149_median-snow.csv'
df_10 = pd.read_csv(df_10_in)

df_25_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\merged_data_products\\merged_uf_r.25m_canopy_19_149_median-snow.csv'
df_25 = pd.read_csv(df_25_in)
df_25.loc[:, 'cc'] = 1 - df_25.loc[:, 'openness']

df_all = pd.concat([df_10, df_25])
# dirty renames
# df_all.loc[:, ['lai_rs', 'lai_hemi', 'hemi_75_deg_tx', 'hemi_15_deg_tx', 'ray_sampled_tx']] = df_all.loc[:, ['cn_mean_25', 'lai_s_cc', 'transmission', 'transmission_1', 'transmission_rs']].values


# combined scatter plots
x_vars = ['swe_19_045', 'swe_19_050', 'swe_19_052', 'dswe_19_045-19_050', 'dswe_19_050-19_052']
y_vars = ['chm_1', 'dnt', 'dce']
hmm = ['chm', 'chm', 'chm', 'chm', 'chm', 'dnt', 'dnt', 'dnt', 'dnt', 'dnt', 'dce', 'dce', 'dce', 'dce', 'dce']
next_label = iter(hmm).__next__
def histme_dce(x, y, color, **kwargs):
    if next_label() == 'dce':
        y_step = 0.1
        plotrange = [[np.nanquantile(x, .0005), np.nanquantile(x, .9995)],
                     np.rint(np.array([np.nanquantile(y, .0005), np.nanquantile(y, .9995)]) / y_step) * y_step]
        n_bins = int((plotrange[1][1] - plotrange[1][0]) / y_step) - 1
        rbins = (8 * 20, n_bins)
    else:
        rbins = (np.array([8, 5.7]) * 20).astype(int)
        plotrange = [[np.nanquantile(x, .0005), np.nanquantile(x, .9995)],
                     [np.nanquantile(y, .0005), np.nanquantile(y, .9995)]]



    # plt.hist2d(x, y, range=plotrange,
    #               bins=rbins, norm=colors.LogNorm(), cmap="Blues")
    plt.hist2d(x, y, range=plotrange, bins=rbins, cmap="Blues")
    # plt.colorbar()
g = sns.PairGrid(df_all, y_vars=y_vars, x_vars=x_vars)
g.map(histme_dce)
# g.fig.subplots_adjust(top=0.9)
# g.fig.suptitle("Scatter plots of snow and canopy metrics", fontsize=16)
plt.savefig(plot_out_dir + "scatter_combined_canopy.png")



def histme(x, y, color, **kwargs):

    rbins = (np.array([8, 5.7]) * 20).astype(int)
    plotrange = [[np.nanquantile(x, .0005), np.nanquantile(x, .9995)],
                 [np.nanquantile(y, .0005), np.nanquantile(y, .9995)]]

    # plt.hist2d(x, y, range=plotrange, bins=rbins, norm=colors.LogNorm(), cmap="Blues")
    plt.hist2d(x, y, range=plotrange, bins=rbins, cmap="Blues")
    # plt.colorbar()

y_vars = ['lai_hemi', 'lai_rs']
g = sns.PairGrid(df_all, y_vars=y_vars, x_vars=x_vars)
g.map(histme)
# g.fig.subplots_adjust(top=0.9)
# g.fig.suptitle("Scatter plots of snow and LAI metrics", fontsize=16)
plt.savefig(plot_out_dir + "scatter_combined_LAI.png")



def histme(x, y, color, **kwargs):
    rbins = (np.array([8, 5.7]) * 20).astype(int)
    # plotrange = [[np.nanquantile(x, .0005), np.nanquantile(x, .9995)],
    #              [np.nanquantile(y, .0005), np.nanquantile(y, .9995)]]
    plotrange = [[np.nanquantile(x, .0005), np.nanquantile(x, .9995)],
                 [0.01, 0.99]]

    # plt.hist2d(x, y, range=plotrange, bins=rbins, norm=colors.LogNorm(), cmap="Blues")
    plt.hist2d(x, y, range=plotrange, bins=rbins, cmap="Blues")
    plt.ylim(0, 1)
    # plt.colorbar()
y_vars = ['hemi_75_deg_tx', 'hemi_15_deg_tx', 'ray_sampled_tx']
g = sns.PairGrid(df_all, y_vars=y_vars, x_vars=x_vars)
g.map(histme)
# g.fig.subplots_adjust(top=0.9)
# g.fig.suptitle("Scatter plots of snow and light transmittance metrics", fontsize=16)
plt.savefig(plot_out_dir + "scatter_combined_transmittance.png")



def histme(x, y, color, **kwargs):
    rbins = (np.array([8, 5.7]) * 20).astype(int)
    # plotrange = [[np.nanquantile(x, .0005), np.nanquantile(x, .9995)],
    #              [np.nanquantile(y, .0005), np.nanquantile(y, .9995)]]

    plotrange = [[np.nanquantile(x, .0005), np.nanquantile(x, .9995)],
                 [0.01, 0.99]]

    # plt.hist2d(x, y, range=plotrange, bins=rbins, norm=colors.LogNorm(), cmap="Blues")
    plt.hist2d(x, y, range=plotrange, bins=rbins, cmap="Blues")
    # plt.colorbar()
y_vars = ['lpmf15', 'lpml15', 'lpmc15']
g = sns.PairGrid(df_all, y_vars=y_vars, x_vars=x_vars)
g.map(histme)
# g.fig.subplots_adjust(top=0.9)
# g.fig.suptitle("Scatter plots of snow and laser penetration metrics", fontsize=16)
plt.savefig(plot_out_dir + "scatter_combined_lpm.png")


# still need colormap...



# individual scatter plots
title_dict = {
    'swe_19_045': '14 Feb. 2019 SWE',
    'swe_19_050': '19 Feb. 2019 SWE',
    'swe_19_052': '21 Feb. 2019 SWE',
    'dswe_19_045-19_050': '14-19 Feb. 2019 $\Delta$SWE',
    'dswe_19_050-19_052': '19-21 Feb. 2019 $\Delta$SWE',
    'chm': 'Canopy Height',
    'dnt': 'Distance to Nearest Tree',
    'dce': 'Distance to Canopy Edge',
    'cn_mean': "Ray Sampled LAI",
    'lai_s_cc': "Hemi-photo LAI",
    'cc': "Canopy Closure",
    'transmission': "Hemispherical Light Transmittance",
    'transmission_rs': "Ray Sampled Light Transmittance",
    'contactnum_1': "15deg Hemi-photo Contact Number",
    'transmission_1': "15deg Hemi-photo light transmittance"}

lab_dict = {
    'swe_19_045': 'SWE [mm]',
    'swe_19_050': 'SWE [mm]',
    'swe_19_052': 'SWE [mm]',
    'dswe_19_045-19_050': '$\Delta$SWE [mm]',
    'dswe_19_050-19_052': '$\Delta$SWE [mm]',
    'chm': 'Canopy height [m]',
    'dnt': 'Distance to Nearest Tree [m]',
    'dce': 'Distance to Canopy Edge [m]',
    'cn_mean': "Ray Sampled LAI [-]",
    'lai_s_cc': "Hemi-photo LAI [-]",
    'cc': "Canopy Closure [-]",
    'transmission': "Hemispherical Light Transmittance [-]",
    'transmission_rs': "Ray Sampled Light Transmittance [-]",
    'contactnum_1': "15deg Hemi-photo Contact Number [-]",
    'transmission_1': "15deg Hemi-photo light transmittance [-]"}

def plot_func(df, x_col, y_col):
    x_dat = df[x_col]
    y_dat = df[y_col]
    filename = "scatter_" + x_col + "_vs_" + y_col + ".png"
    x_lab = lab_dict[x_col]
    y_lab = lab_dict[y_col]
    title = title_dict[x_col] + " vs. " + title_dict[y_col] + '\nUpper Forest, 10cm resolution'

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    if y_col == "dce":
        y_step = 0.1
        plotrange = [[np.nanquantile(x_dat, .0005), np.nanquantile(x_dat, .9995)],
                     np.rint(np.array([np.nanquantile(y_dat, .0005), np.nanquantile(y_dat, .9995)]) / y_step) * y_step]
        n_bins = int((plotrange[1][1] - plotrange[1][0]) / y_step) - 1
        h = ax.hist2d(x_dat, y_dat, range=plotrange,
                      bins=(8 * 20, n_bins), norm=colors.LogNorm(), cmap="Blues")
    else:
        plotrange = [[np.nanquantile(x_dat, .0005), np.nanquantile(x_dat, .9995)],
                     [np.nanquantile(y_dat, .0005), np.nanquantile(y_dat, .9995)]]
        h = ax.hist2d(x_dat, y_dat, range=plotrange,
                      bins=(np.array([8, 5.7]) * 20).astype(int), norm=colors.LogNorm(), cmap="Blues")
    fig.colorbar(h[3], ax=ax)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(title)
    fig.savefig(plot_out_dir + filename)

x_vars = ['swe_19_045', 'swe_19_050', 'swe_19_052', 'dswe_19_045-19_050', 'dswe_19_050-19_052']
y_vars = ['chm', 'dnt', 'cn_mean', 'lai_s_cc', 'cc', 'transmission', 'transmission_rs', 'contactnum_1', 'transmission_1', 'dce']


for xx in x_vars:
    for yy in y_vars:
        plot_func(df_all, xx, yy)




# spearmans heatmap

x_vars = ['swe_fcon_19_045', 'swe_fcon_19_050', 'swe_fcon_19_052', 'dswe_fnsd_19_045-19_050', 'dswe_fnsd_19_050-19_052']
x_labs = ['SWE\n045', 'SWE\n050', 'SWE\n052', '$\Delta$SWE\nStorm 1', '$\Delta$SWE\nStorm 2']

y_vars = ['dnt', 'dce', 'chm',
          'mCH_19_149_resampled', 'mCH_045_050_052_resampled', 'mCH_19_149',
          'fcov', 'lpml15',
          'lrs_cn_1', 'lrs_cn_1_snow_on', 'contactnum_1',
          'lrs_cn_5', 'lrs_cn_5_snow_on', 'contactnum_5',
          'lrs_lai_75_deg', 'lrs_lai_75_deg_snow_on',
          'lrs_lai_2000', 'lrs_lai_2000_snow_on', 'lai_no_cor',
          'lrs_cc', 'lrs_cc_snow_on', 'cc'
          ]
y_labs = ['$DNT$', '$DCE$', '$CHM$',
          '$mCH$', '$mCH^{*}$', '$mCH^{\dagger}$',
          '$fCov$', '$LPM$-$L$',
          '$Cn_{15}$', '$Cn_{15}^{*}$', '$Cn_{15}^{\dagger}$',
          '$Cn_{60-75}$', '$Cn_{60-75}^{*}$', '$Cn_{60-75}^{\dagger}$',
          '$LAI_{75}$', '$LAI_{75}^{*}$',
          '$LAI_{2000}$', '$LAI_{2000}^{*}$', '$LAI_{2000}^{\dagger}$',
          '$CC$', '$CC^{*}$', '$CC^{\dagger}$'
          ]

# export stats
spr = np.full((len(y_vars), len(x_vars)), np.nan)
spr_p = np.full((len(y_vars), len(x_vars)), np.nan)
df = np.full((len(y_vars), len(x_vars)), np.nan)
for ii in range(len(x_vars)):
    xx = x_vars[ii]
    for jj in range(len(y_vars)):
        yy = y_vars[jj]

        valid = ~np.isnan(df_all[xx]) & ~np.isnan(df_all[yy])

        df[jj, ii] = np.sum(valid) - 2
        spr[jj, ii], spr_p[jj, ii] = spearmanr(df_all.loc[valid, xx], df_all.loc[valid, yy])

stat_file = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\covariate_stats\\'
np.savetxt(stat_file + "spearmans_r.csv", spr, delimiter=", ")
np.savetxt(stat_file + "spearmans_r_p-values.csv", spr_p, delimiter=", ")
np.savetxt(stat_file + "degrees_of_freedom.csv", df, delimiter=", ")

pp = spr_p.astype(str)

spr_str = spr.astype(str)
imax, jmax = spr.shape
for ii in range(0, imax):
    for jj in range(0, jmax):
        if spr_p[ii, jj] < 0.001:
            # spr_str[ii, jj] = "{0:3.3f}\n(<.001)".format(spr[ii, jj])
            spr_str[ii, jj] = "{0:3.3f}".format(spr[ii, jj])
        elif spr_p[ii, jj] >= 0.05:
            spr_str[ii, jj] = "{0:3.3f}**\n({1:3.3f})".format(spr[ii, jj], spr_p[ii, jj])
        else:
            spr_str[ii, jj] = "{0:3.3f}\n({1:3.3f})".format(spr[ii, jj], spr_p[ii, jj])

# plot heatmap
correlation_mat = df_all.loc[:, x_vars + y_vars].corr(method="spearman")
correlation_mat_sub = correlation_mat.loc[y_vars, x_vars]
fig, ax = plt.subplots(figsize=(6, 8), constrained_layout=True)
sns.heatmap(correlation_mat_sub, annot=spr_str, fmt='', vmin=-1, vmax=1, cmap="RdBu", yticklabels=y_labs, xticklabels=x_labs)
fig.savefig(plot_out_dir + "spearman_heatmap.png")




# # cross scatters
# import rastools
#
# ddict = {'uf': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\hemi_grid_points\\mb_65_1m\\uf_plot_r.25m.tif',
#         ('er_p0_mean', 'er_p0_sd'): 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_mb_15_dem_.25m_61px_mp15.25\\outputs\\las_19_149_rs_mb_15_r.25_p0.0000_t3.1416.tif',
#         'lpmf15': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\LPM\\19_149_LPM-first_a15_r0.10m.tif'}
# xc = rastools.pd_sample_raster_gdal(ddict, include_nans=True, mode="median")
# xc = xc.loc[xc.uf == 1, :]
# xc.loc[:, 'cn_mean'] = xc.loc[:, 'er_p0_mean'] * 0.19447
# xc.loc[:, 'transmission_rs'] = np.exp(-xc.loc[:, 'cn_mean'])


x_dat = df_25.lai_s_cc
y_dat = df_25.lrs_cn_75_deg

x_dat = df_25.swe_fcon_19_050
y_dat = df_25.lrs_tx_1

x_dat = df_25.transmission_s_1
y_dat = df_25.lrs_tx_1

x_dat = df_25.lai_no_cor
y_dat = df_25.lrs_lai_2000

plotrange = [[np.nanquantile(x_dat, .0005), np.nanquantile(x_dat, .9995)],
                     [np.nanquantile(y_dat, .0005), np.nanquantile(y_dat, .9995)]]
plotrange = [[0, np.nanquantile(x_dat, .9995)],
                     [0, np.nanquantile(y_dat, .9995)]]
# plotrange = [[0, np.nanquantile(x_dat, .9995)],
#                      [0, 4]]
# plt.hist2d(x_dat, y_dat, range=plotrange, bins=(np.array([8, 5.7]) * 20).astype(int), norm=colors.LogNorm(), cmap="Blues")
plt.hist2d(x_dat, y_dat, range=plotrange, bins=(np.array([8, 5.7]) * 20).astype(int), cmap="Blues")
fig, ax = plt.subplots(nrows=2, ncols=2)



def plot_together(df, x_dat, y_dat, titles, suptitle="", plotrange=None):
    n_plots = len(y_dat)
    fig, ax = plt.subplots(nrows=1, ncols=n_plots, sharey=True, sharex=True, figsize=(3 * n_plots, 3.8), constrained_layout=True)

    if plotrange is None:
        plotrange = [[0, np.nanquantile(df.loc[:, x_dat], .9995)],
                     [0, np.nanquantile(df.loc[:, y_dat], .9995)]]
    squarerange = [[np.min(plotrange), np.max(plotrange)],
                   [np.min(plotrange), np.max(plotrange)]]
    xx = [np.min(plotrange), np.max(plotrange)]
    for ii in range(0, n_plots):
        # 1-1 line
        ax[ii].plot(xx, xx, color="black", linewidth=0.5)

        ax[ii].hist2d(df.loc[:, x_dat[ii]], df.loc[:, y_dat[ii]], range=squarerange, bins=(np.array([8, 5.7]) * 20).astype(int), cmap="Blues")
        ax[ii].title.set_text(titles[ii])
        if ii == 0:
            ax[ii].set_ylabel("Ray sampling contact number [-]")
        if ii == 2:
            ax[ii].set_xlabel("Point reprojection contact number [-]")

    fig.add_subplot(111, frameon=False)
    if suptitle is not "":
        plt.suptitle(suptitle)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    # plt.xlabel("Lidar point reprojection")
    # plt.ylabel("Lidar ray sampling")

    return fig, ax

x_dat = ["contactnum_1",
         "contactnum_2",
         "contactnum_3",
         "contactnum_4",
         "contactnum_5"]

y_dat = ["lrs_cn_1",
         "lrs_cn_2",
         "lrs_cn_3",
         "lrs_cn_4",
         "lrs_cn_5"]

titles = ["0-15$^{\circ}$",
          "15-30$^{\circ}$",
          "30-45$^{\circ}$",
          "45-60$^{\circ}$",
          "60-75$^{\circ}$"]


fig, ax = plot_together(df_25, x_dat, y_dat, titles, suptitle="Comparison of contact number between methods across angle bands for Upper Forest")
fig.savefig(plot_out_dir + "cn_comparison.png")

x_dat = ["transmission_s_1",
         "transmission_s_2",
         "transmission_s_3",
         "transmission_s_4",
         "transmission_s_5"]

y_dat = ["lrs_tx_1",
         "lrs_tx_2",
         "lrs_tx_3",
         "lrs_tx_4",
         "lrs_tx_5"]

titles = ["0-15$^{\circ}$",
          "15-30$^{\circ}$",
          "30-45$^{\circ}$",
          "45-60$^{\circ}$",
          "60-75$^{\circ}$"]


fig, ax = plot_together(df_25, x_dat, y_dat, titles, suptitle="Comparison of transmittance between methods across angle bands for Upper Forest")
fig.savefig(plot_out_dir + "tx_comparison.png")

fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True, sharex=True, figsize=(8, 8), constrained_layout=True)
x_dat = ["lai_no_cor"]
y_dat = ["lrs_lai_75_deg"]
titles = ["LAI comparison between methods across angle bands for Upper Forest"]
df = df_25
ii = 0
x = [0, 100]
y = [0, 100]
plt.plot(x, y, color="black", linewidth=0.5)
offset = 0
plotrange = [[np.nanquantile(df.loc[:, x_dat], .0005) - offset, np.nanquantile(df.loc[:, x_dat], .995) + offset],
                 [np.nanquantile(df.loc[:, y_dat], .0005) - offset, np.nanquantile(df.loc[:, y_dat], .995) + offset]]
squarerange = [[np.min(plotrange), np.max(plotrange)],
               [np.min(plotrange), np.max(plotrange)]]
ax.hist2d(df.loc[:, x_dat[ii]], df.loc[:, y_dat[ii]], range=squarerange,
          bins=(np.array([8, 8]) * 20).astype(int), cmap="Blues", norm=colors.LogNorm())
ax.title.set_text(titles[ii])
ax.set_ylabel("Ray sampling LAI [-]")
ax.set_xlabel("Point reprojection LAI [-]")
fig.savefig(plot_out_dir + "lai_comparison.png")

# x_dat = ["lai_no_cor",
#          "lai_no_cor",
#          "lai_no_cor"]
#
# y_dat = ["lrs_lai_15_deg",
#          "lrs_lai_75_deg",
#          "lrs_lai_1_deg"]
#
# titles = ["LAI 15$^{\circ}$",
#           "LAI 75$^{\circ}$",
#           "LAI 1$^{\circ}$"]
#
# fig, ax = plot_together(df_25, x_dat, y_dat, titles, suptitle="LAI comparison between methods across angle bands for Upper Forest", plotrange = [[0, 5], [0, 5]])
# fig.savefig(plot_out_dir + "lai_comparison.png")

xx = "lrs_lai_75_deg"
# yy = "lrs_lai_15_deg"
yy = "lrs_lai_1_deg"
valid = ~np.isnan(df_25[xx]) & ~np.isnan(df_25[yy])
spearmanr(df_25.loc[valid, xx], df_25.loc[valid, yy])
pearsonr(df_25.loc[valid, xx], df_25.loc[valid, yy])