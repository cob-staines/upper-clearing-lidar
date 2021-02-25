import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns

plot_out_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\graphics\\thesis_graphics\\scatter plots\\"

df_10_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\uf_merged_.10m_ahpl_median_canopy_19_149.csv'
df_10 = pd.read_csv(df_10_in)
df_10 = df_10.loc[df_10.uf == 1, :]

df_25_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\mb_15_merged_.25m_ahpl_median_canopy_19_149.csv'
df_25 = pd.read_csv(df_25_in)
df_25 = df_25.loc[df_25.uf == 1, :]
df_25.loc[:, 'cn_mean'] = df_25.loc[:, 'er_p0_mean'] * 0.19447
df_25.loc[:, 'transmission_rs'] = np.exp(-df_25.loc[:, 'cn_mean'])
df_25.loc[:, 'cc'] = 1 - df_25.loc[:, 'openness']

df_all = pd.concat([df_10, df_25])
# dirty renames
df_all.loc[:, ['lai_rs', 'lai_hemi', 'hemi_75_deg_tx', 'hemi_15_deg_tx', 'ray_sampled_tx']] = df_all.loc[:, ['cn_mean', 'lai_s_cc', 'transmission', 'transmission_1', 'transmission_rs']].values


# combined scatter plots


x_vars = ['swe_19_045', 'swe_19_050', 'swe_19_052', 'dswe_19_045-19_050', 'dswe_19_050-19_052']
y_vars = ['chm', 'dnt', 'dce']
hmm = ['chm', 'chm', 'chm', 'chm', 'chm', 'dnt', 'dnt', 'dnt', 'dnt', 'dnt', 'dce', 'dce', 'dce', 'dce', 'dce']
next_label = iter(hmm).__next__
def histme(x, y, color, **kwargs):
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



    plt.hist2d(x, y, range=plotrange,
                  bins=rbins, norm=colors.LogNorm(), cmap="Blues")
    # plt.colorbar()
g = sns.PairGrid(df_all, y_vars=y_vars, x_vars=x_vars)
g.map(histme)
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Scatter plots of snow and canopy metrics", fontsize=16)
plt.savefig(plot_out_dir + "scatter_combined_canopy.png")

hmm = ['chm', 'chm', 'chm', 'chm', 'chm', 'dnt', 'dnt', 'dnt', 'dnt', 'dnt', 'nope', 'nope', 'nope', 'nope', 'nope']
next_label = iter(hmm).__next__
y_vars = ['lai_hemi', 'lai_rs']

def histme(x, y, color, **kwargs):
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



    plt.hist2d(x, y, range=plotrange,
                  bins=rbins, norm=colors.LogNorm(), cmap="Blues")
    # plt.colorbar()
g = sns.PairGrid(df_all, y_vars=y_vars, x_vars=x_vars)
g.map(histme)
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Scatter plots of snow and LAI metrics", fontsize=16)
plt.savefig(plot_out_dir + "scatter_combined_LAI.png")


hmm = ['chm', 'chm', 'chm', 'chm', 'chm', 'dnt', 'dnt', 'dnt', 'dnt', 'dnt', 'nope', 'nope', 'nope', 'nope', 'nope']
next_label = iter(hmm).__next__
y_vars = ['hemi_75_deg_tx', 'hemi_15_deg_tx', 'ray_sampled_tx']
def histme(x, y, color, **kwargs):
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



    plt.hist2d(x, y, range=plotrange,
                  bins=rbins, norm=colors.LogNorm(), cmap="Blues")
    # plt.colorbar()
g = sns.PairGrid(df_all, y_vars=y_vars, x_vars=x_vars)
g.map(histme)
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Scatter plots of snow and light transmittance metrics", fontsize=16)
plt.savefig(plot_out_dir + "scatter_combined_transmittance.png")

hmm = ['chm', 'chm', 'chm', 'chm', 'chm', 'dnt', 'dnt', 'dnt', 'dnt', 'dnt', 'nope', 'nope', 'nope', 'nope', 'nope']
next_label = iter(hmm).__next__
y_vars = ['lpmf15', 'lpml15', 'lpmc15']
def histme(x, y, color, **kwargs):
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



    plt.hist2d(x, y, range=plotrange,
                  bins=rbins, norm=colors.LogNorm(), cmap="Blues")
    # plt.colorbar()
g = sns.PairGrid(df_all, y_vars=y_vars, x_vars=x_vars)
g.map(histme)
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Scatter plots of snow and laser penetration metrics", fontsize=16)
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
#
# x_dat = xc.lpmf15
# y_dat = xc.transmission_rs
# plotrange = [[np.nanquantile(x_dat, .0005), np.nanquantile(x_dat, .9995)],
#                      [np.nanquantile(y_dat, .0005), np.nanquantile(y_dat, .9995)]]
# plt.hist2d(x_dat, y_dat, range=plotrange, bins=(np.array([8, 5.7]) * 20).astype(int), norm=colors.LogNorm(), cmap="Blues")
