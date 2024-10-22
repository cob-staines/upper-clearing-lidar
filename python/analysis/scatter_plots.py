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

df_25_ad_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\merged_data_products\\merged_uf_r.25m_canopy_19_149_median-snow_alldeg.csv'
df_25_ad = pd.read_csv(df_25_ad_in)

print(set(df_10.columns) & set(df_25.columns))  # make sure no covariates (canopy metrics) show up in this list!
df_all = pd.concat([df_10, df_25])
# dirty renames
# df_all.loc[:, ['lai_rs', 'lai_hemi', 'hemi_75_deg_tx', 'hemi_15_deg_tx', 'ray_sampled_tx']] = df_all.loc[:, ['cn_mean_25', 'lai_s_cc', 'transmission', 'transmission_1', 'transmission_rs']].values


# combined scatter plots
x_vars = ['swe_fcon_19_045', 'swe_fcon_19_050', 'swe_fcon_19_052', 'dswe_fnsd_19_045-19_050', 'dswe_fnsd_19_050-19_052']
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
    'swe_fcon_19_045': '14 Feb. 2019 SWE',
    'swe_fcon_19_050': '19 Feb. 2019 SWE',
    'swe_fcon_19_052': '21 Feb. 2019 SWE',
    'dswe_fnsd_19_045-19_050': '14-19 Feb. 2019 $\Delta$SWE',
    'dswe_fnsd_19_050-19_052': '19-21 Feb. 2019 $\Delta$SWE',
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
    'swe_fcon_19_045': 'SWE [mm]',
    'swe_fcon_19_050': 'SWE [mm]',
    'swe_fcon_19_052': 'SWE [mm]',
    'dswe_fnsd_19_045-19_050': '$\Delta$SWE [mm]',
    'dswe_fnsd_19_050-19_052': '$\Delta$SWE [mm]',
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

x_vars = ['swe_fcon_19_045', 'swe_fcon_19_050', 'swe_fcon_19_052', 'dswe_fnsd_19_045-19_050', 'dswe_fnsd_19_050-19_052']
y_vars = ['chm', 'dnt', 'cn_mean', 'lai_s_cc', 'cc', 'transmission', 'transmission_rs', 'contactnum_1', 'transmission_1', 'dce']


for xx in x_vars:
    for yy in y_vars:
        plot_func(df_all, xx, yy)




# spearmans heatmap


x_vars = ['swe_fcon_19_045', 'swe_fcon_19_050', 'swe_fcon_19_052', 'dswe_fnsd_19_045-19_050', 'dswe_fnsd_19_050-19_052']
x_labs = ['SWE\nFeb 14', 'SWE\nFeb 19', 'SWE\nFeb 21', '$\Delta$SWE\nStorm 1', '$\Delta$SWE\nStorm 2']
#
# y_vars = ['chm', 'dnt', 'dce',
#           'mCH_19_149_resampled', 'fcov', 'lpml15',
#           'mean_dist_to_canopy_1.25', 'total_gap_area_1.25',
#           # 'lrs_cn_1_deg', 'lrs_cn_1_deg_snow_on',
#           'lrs_tx_1_deg_snow_on', 'lrs_tx_1_deg',
#           # 'lrs_cn_1', 'lrs_cn_1_snow_on', 'contactnum_1',
#           'lrs_tx_1_snow_on', 'lrs_tx_1', 'transmission_s_1',
#           # 'lrs_cn_5', 'lrs_cn_5_snow_on', 'contactnum_5',
#           'lrs_tx_2_snow_on', 'lrs_tx_2', 'transmission_s_2',
#           'lrs_lai_60_deg_snow_on', 'lrs_lai_60_deg', 'lai_s_cc',
#           'lrs_cc_snow_on', 'lrs_cc', 'cc',
#           'lrs_sky_view_snow_on', 'lrs_sky_view'
#           ]
# y_labs = ['$CHM$', '$DNT$', '$DCE$',
#           r'$MCH$', '$fCov$', '$LPM$-$L$',
#           r'$MDC$', r'$TGA$',
#           # r'$\chi_{1}^{\blacktriangle}$', r'$\chi_{1}^{\vartriangle}$',
#           r'$T_{1}^{\vartriangle}$', r'$T_{1}^{\blacktriangle}$',
#           # r'$\chi_{15}^{\blacktriangle}$', r'$\chi_{15}^{\vartriangle}$', r'$\chi_{15}^{\bullet}$',
#           r'$T_{15}^{\vartriangle}$', r'$T_{15}^{\blacktriangle}$', r'$T_{15}^{\bullet}$',
#           # r'$\chi_{60-75}^{\blacktriangle}$', r'$\chi_{60-75}^{\vartriangle}$', r'$\chi_{60-75}^{\bullet}$',
#           r'$T_{15-30}^{\vartriangle}$', r'$T_{15-30}^{\blacktriangle}$', r'$T_{15-30}^{\bullet}$',
#           r'$LAI_{60}^{\vartriangle}$', r'$LAI_{60}^{\blacktriangle}$', r'$LAI_{2000}^{\bullet}$',
#           r'$C_{c}^{\vartriangle}$', r'$C_{c}^{\blacktriangle}$', r'$C_{c}^{\bullet}$',
#           r'$V_{f}^{\vartriangle}$', r'$V_{f}^{\blacktriangle}$'
#           ]

# manuscript
y_vars = ['chm',
          'dnt',
          'dce',
          'lpml15',
          'mean_dist_to_canopy_1.25',
          'total_gap_area_1.25',
          # 'lrs_cn_1_deg', 'lrs_cn_1_deg_snow_on',
          # 'lrs_tx_1_deg_snow_on', 'lrs_tx_1_deg',
          # 'lrs_cn_1', 'lrs_cn_1_snow_on', 'contactnum_1',
          # 'lrs_tx_1_snow_on', 'lrs_tx_1',
          # 'lrs_cn_5', 'lrs_cn_5_snow_on', 'contactnum_5',
          # 'lrs_tx_2_snow_on', 'lrs_tx_2',
          'lrs_lai_1_deg_snow_on',
          'lrs_lai_15_deg_snow_on',
          'lrs_lai_60_deg_snow_on',
          'lrs_lai_2000_snow_on',
          'lrs_lai_2000',
          # 'lrs_lai_60_deg',
          'lrs_cc_snow_on', 'lrs_cc',
          # 'lrs_sky_view_snow_on', 'lrs_sky_view'
          ]
y_labs = ['$CHM$',
          '$DNT$',
          '$DCE$',
          '$LPM$-$L$',
          r'$MDC$',
          r'$TGA$',
          # r'$\chi_{1}^{\blacktriangle}$', r'$\chi_{1}^{\vartriangle}$',
          # r'$T_{1}^{\vartriangle}$', r'$T_{1}^{\blacktriangle}$',
          # r'$\chi_{15}^{\blacktriangle}$', r'$\chi_{15}^{\vartriangle}$', r'$\chi_{15}^{\bullet}$',
          # r'$T_{15}^{\vartriangle}$', r'$T_{15}^{\blacktriangle}$',
          # r'$\chi_{60-75}^{\blacktriangle}$', r'$\chi_{60-75}^{\vartriangle}$', r'$\chi_{60-75}^{\bullet}$',
          # r'$T_{15-30}^{\vartriangle}$', r'$T_{15-30}^{\blacktriangle}$',
          r'$LAI_{1}^{\vartriangle}$',
          r'$LAI_{15}^{\vartriangle}$',
          r'$LAI_{60}^{\vartriangle}$',
          r'$LAI_{2000}^{\vartriangle}$',
          r'$LAI_{2000}^{\blacktriangle}$',
          # r'$LAI_{60}^{\blacktriangle}$',
          r'$C_{c}^{\vartriangle}$', r'$C_{c}^{\blacktriangle}$',
          # r'$V_{f}^{\vartriangle}$', r'$V_{f}^{\blacktriangle}$'
          ]

# # full lables
# y_labs = [r'Canopy crown height - $CHM$ (m AGS)', r'Distance to nearest tree - $DNT$ (m)', r'Distance to canopy edge - $DCE$ (m)',
#           r'Mean canopy height - $mCH$ (m AGS)', 'Fractional cover - $fCov$ (-)', 'Laser penetration metric $LPM$-$L$ (-)',
#           r'Mean distance to canopy - $MDC$ (m)', r'Total gap area - $TGA$ ($m^2$)',
#           r'Light transmittance ($1^\circ$, ray sampling, snow on) - $T_{1}^{\vartriangle}$ (-)', r'Light transmittance ($1^\circ$, ray sampling, snow off) - $T_{1}^{\blacktriangle}$ (-)',
#           # r'$\chi_{15}^{\blacktriangle}$', r'$\chi_{15}^{\vartriangle}$', r'$\chi_{15}^{\bullet}$',
#           r'Light transmittance ($15^\circ$, ray sampling, snow on) - $T_{15}^{\vartriangle}$ (-)', r'Light transmittance ($15^\circ$, ray sampling, snow off) - $T_{15}^{\blacktriangle}$ (-)', r'Light transmittance ($15^\circ$, reprojection, snow off) - $T_{15}^{\bullet}$ (-)',
#           # r'$\chi_{60-75}^{\blacktriangle}$', r'$\chi_{60-75}^{\vartriangle}$', r'$\chi_{60-75}^{\bullet}$',
#           r'Light transmittance ($15^\circ$ - $30^\circ$, ray sampling, snow on) - $T_{15-30}^{\vartriangle}$ (-)', r'Light transmittance ($15^\circ$ - $30^\circ$, ray sampling, snow off) - $T_{15-30}^{\blacktriangle}$ (-)', r'Light transmittance ($15^\circ$ - $30^\circ$, reprojection, snow off) - $T_{15-30}^{\bullet}$ (-)',
#           r'Leaf area index ($60^\circ$, ray sampling, snow on) - $LAI_{60}^{\vartriangle}$ (-)', r'Leaf area index ($60^\circ$, ray sampling, snow off) - $LAI_{60}^{\blacktriangle}$ (-)', r'Leaf area index ($60^\circ$, reprojection, snow off) - $LAI_{2000}^{\bullet}$ (-)',
#           r'Canopy closure ($60^\circ$, ray sampling, snow on) - $C_{c}^{\vartriangle}$ (-)', r'Canopy closure ($60^\circ$, ray sampling, snow off) - $C_{c}^{\blacktriangle}$ (-)', r'Canopy closure ($60^\circ$, reprojection, snow off) - $C_{c}^{\bullet}$ (-)',
#           r'Sky view factor ($60^\circ$, ray sampling, snow on) - $V_{f}^{\vartriangle}$ (-)', r'Sky view factor ($60^\circ$, ray sampling, snow off) - $V_{f}^{\blacktriangle}$ (-)',
#           ]

y_res = [10,
         10,
         10,
         25,
         25,
         25,
         25,
         25,
         25,
         25,
         25, 25, 25
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
np.savetxt(stat_file + "spearmans_r_manuscript.csv", spr, delimiter=", ")
np.savetxt(stat_file + "spearmans_r_p-values_manuscript.csv", spr_p, delimiter=", ")
np.savetxt(stat_file + "degrees_of_freedom.csv_manuscript", df, delimiter=", ")

pp = spr_p.astype(str)

spr_str = spr.astype(str)
imax, jmax = spr.shape
for ii in range(0, imax):
    for jj in range(0, jmax):
        if spr_p[ii, jj] < 0.001:
            # spr_str[ii, jj] = "{0:3.3f}\n(<.001)".format(spr[ii, jj])
            spr_str[ii, jj] = "{0:3.3f}$^*$".format(spr[ii, jj])
        elif spr_p[ii, jj] >= 0.05:
            spr_str[ii, jj] = "{0:3.3f}\n({1:3.3f})".format(spr[ii, jj], spr_p[ii, jj])
        else:
            spr_str[ii, jj] = "{0:3.3f}$^*$\n({1:3.3f})".format(spr[ii, jj], spr_p[ii, jj])

# plot heatmap

correlation_mat = df_all.loc[:, x_vars + y_vars].corr(method="spearman")
correlation_mat_sub = correlation_mat.loc[y_vars, x_vars]
fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
sns.heatmap(correlation_mat_sub, annot=spr_str, fmt='', vmin=-1, vmax=1, cmap="RdBu", yticklabels=y_labs, xticklabels=x_labs, cbar_kws={'label': "Spearnam's Rank Correlation Coefficient"})
fig.savefig(plot_out_dir + "spearman_heatmap_manuscript.png")




# calculate medians of canopy metrics
# all points

y_stats = ["mean_all",
           "std_all",
           "median_all",
           "median_canopy",
           "max_all"]

can_stats = pd.DataFrame(columns=y_stats, index=y_vars, data=np.nan)

canopy_10 = (df_all.chm >= 1)
canopy_25 = (df_all.chm_median >= 1)
for jj in range(len(y_vars)):
        yy = y_vars[jj]
        can_stats.mean_all[jj] = np.nanmean(df_all.loc[:, yy])
        can_stats.std_all[jj] = np.nanstd(df_all.loc[:, yy])
        can_stats.median_all[jj] = np.nanmedian(df_all.loc[:, yy])
        can_stats.max_all[jj] = np.nanmax(df_all.loc[:, yy])

        if y_res[jj] == 10:
            canbool = canopy_10
        elif y_res[jj] == 25:
            canbool = canopy_25

        can_stats.median_canopy[jj] = np.nanmedian(df_all.loc[canbool, yy])
        # can_stats.max_canopy[jj] = np.nanmax(df_all.loc[canbool, yy])

can_stats.to_csv(stat_file + "canopy_stats_manuscript.csv")



##############################################################


# Plot tranmittance correlations across angles
x_vars = ['swe_fcon_19_045', 'swe_fcon_19_050', 'swe_fcon_19_052', 'dswe_ucgo_19_045-19_050', 'dswe_ucgo_19_050-19_052']
x_labs = ['SWE\nFeb 14', 'SWE\nFeb 19', 'SWE\nFeb 21', '$\Delta$SWE\nStorm 1', '$\Delta$SWE\nStorm 2']

y_vars_snow_off = []
for dd in range(1, 91):
    y_vars_snow_off.append("lrs_tx_" + str(dd) + "_deg")

y_vars_snow_on = []
for dd in range(1, 91):
    y_vars_snow_on.append("lrs_tx_" + str(dd) + "_deg_snow_on")

spr_snow_on = np.full((len(y_vars_snow_on), len(x_vars)), np.nan)
spr_p_snow_on = np.full((len(y_vars_snow_on), len(x_vars)), np.nan)

spr_snow_off = np.full((len(y_vars_snow_off), len(x_vars)), np.nan)
spr_p_snow_off = np.full((len(y_vars_snow_off), len(x_vars)), np.nan)
for ii in range(len(x_vars)):
    xx = x_vars[ii]
    for jj in range(0, 90):
        yy_snow_off = y_vars_snow_off[jj]
        valid = ~np.isnan(df_25_ad[xx]) & ~np.isnan(df_25_ad[yy_snow_off])
        spr_snow_off[jj, ii], spr_p_snow_off[jj, ii] = spearmanr(df_25_ad.loc[valid, xx], df_25_ad.loc[valid, yy_snow_off])

        yy_snow_on = y_vars_snow_on[jj]
        valid = ~np.isnan(df_25_ad[xx]) & ~np.isnan(df_25_ad[yy_snow_on])
        spr_snow_on[jj, ii], spr_p_snow_on[jj, ii] = spearmanr(df_25_ad.loc[valid, xx], df_25_ad.loc[valid, yy_snow_on])

fig = plt.figure()
axes = fig.add_axes([0.13, 0.1, 0.62, 0.8])
# adding axes
plt.xlabel('Max angle from zenith [deg]')
plt.ylabel('Spearmans Rank Corrolation Coefficient [-]')
# plt.title('Relative variance of SWE metrics with distance\n Upper Forest, 5cm resolution (n=1200000)')

degmat = np.linspace(1, 90, 90)

axes.plot(degmat, spr_snow_off[:, 0], label=x_labs[0] + "\n(snow off)", color='C0')
axes.plot(degmat, spr_snow_off[:, 1], label=x_labs[1] + "\n(snow off)", color='C1')
axes.plot(degmat, spr_snow_off[:, 2], label=x_labs[2] + "\n(snow off)", color='C2')

axes.plot(degmat, spr_snow_on[:, 0], label=x_labs[0] + "\n(snow on)", linestyle="dashed", color='C0')
axes.plot(degmat, spr_snow_on[:, 1], label=x_labs[1] + "\n(snow on)", linestyle="dashed", color='C1')
axes.plot(degmat, spr_snow_on[:, 2], label=x_labs[2] + "\n(snow on)", linestyle="dashed", color='C2')

axes.legend(loc='center left', bbox_to_anchor=(1.01, .5), ncol=1,
            borderaxespad=0, frameon=False)

fig = plt.figure()
axes = fig.add_axes([0.13, 0.1, 0.62, 0.8])
# adding axes
plt.xlabel('Angular footprint of light transmittance [deg]')
plt.ylabel('Spearmans Rank Corrolation Coefficient [-]\n between canopy light transmittance and snow metrics')
# plt.title('Relative variance of SWE metrics with distance\n Upper Forest, 5cm resolution (n=1200000)')

degmat = np.linspace(1, 90, 90)

axes.plot(degmat, spr_snow_on[:, 0], label=x_labs[0] + "\n(snow on)", c="C0")
axes.plot(degmat, spr_snow_off[:, 0], label="(snow off)", linestyle="dashed", c="C0")
axes.plot(degmat, spr_snow_on[:, 1], label=x_labs[1] + "\n(snow on)", c="C1")
axes.plot(degmat, spr_snow_off[:, 1], label="(snow off)", linestyle="dashed", c="C1")
axes.plot(degmat, spr_snow_on[:, 2], label=x_labs[2] + "\n(snow on)", c="C2")
axes.plot(degmat, spr_snow_off[:, 2], label="(snow off)", linestyle="dashed", c="C2")
axes.plot(degmat, spr_snow_on[:, 3], label=x_labs[3] + "\n(snow on)", c="C3")
axes.plot(degmat, spr_snow_off[:, 3], label="(snow off)", linestyle="dashed", c="C3")
axes.plot(degmat, spr_snow_on[:, 4], label=x_labs[4] + "\n(snow on)", c="C4")
axes.plot(degmat, spr_snow_off[:, 4], label="(snow off)", linestyle="dashed", c="C4")

axes.legend(loc='center left', bbox_to_anchor=(1.01, .5), ncol=1,
            borderaxespad=0, frameon=False)
axes.set_ylim(0,)
fig.savefig(plot_out_dir + "spearmans_angular_footprint.png")


# Determine maxima and location
np.max(spr_snow_on[:, 0])
degmat[np.argmax(spr_snow_on[:, 0])]

np.max(spr_snow_on[:, 1])
degmat[np.argmax(spr_snow_on[:, 1])]

np.max(spr_snow_on[:, 2])
degmat[np.argmax(spr_snow_on[:, 2])]

np.max(spr_snow_on[:, 3])
degmat[np.argmax(spr_snow_on[:, 3])]

np.max(spr_snow_on[:, 4])
degmat[np.argmax(spr_snow_on[:, 4])]



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
#
# x_dat = df_25.lai_s_cc
# y_dat = df_25.lrs_cn_75_deg
#
# x_dat = df_25.swe_fcon_19_050
# y_dat = df_25.lrs_tx_1
#
# x_dat = df_25.transmission_s_1
# y_dat = df_25.lrs_tx_1
#
# x_dat = df_25.lai_no_cor
# y_dat = df_25.lrs_lai_2000
#
# plotrange = [[np.nanquantile(x_dat, .0005), np.nanquantile(x_dat, .9995)],
#                      [np.nanquantile(y_dat, .0005), np.nanquantile(y_dat, .9995)]]
# plotrange = [[0, np.nanquantile(x_dat, .9995)],
#                      [0, np.nanquantile(y_dat, .9995)]]
# # plotrange = [[0, np.nanquantile(x_dat, .9995)],
# #                      [0, 4]]
# # plt.hist2d(x_dat, y_dat, range=plotrange, bins=(np.array([8, 5.7]) * 20).astype(int), norm=colors.LogNorm(), cmap="Blues")
# plt.hist2d(x_dat, y_dat, range=plotrange, bins=(np.array([8, 5.7]) * 20).astype(int), cmap="Blues")
# fig, ax = plt.subplots(nrows=2, ncols=2)



def plot_together(df, x_dat, y_dat, titles, lims, suptitle="", x_lab=None, y_lab=None, x_weights=None):
    n_plots = len(y_dat)
    fig, ax = plt.subplots(nrows=1, ncols=n_plots, sharey=True, sharex=True, figsize=(3 * n_plots, 3.8), constrained_layout=True)

    mb = []
    minmax = [np.min(lims), np.max(lims)]

    for ii in range(0, n_plots):
        # 1-1 line
        ax[ii].plot(minmax, minmax, color="black", linewidth=0.5)

        # ax[ii].hist2d(df.loc[:, x_dat[ii]], df.loc[:, y_dat[ii]], range=squarerange, bins=(np.array([8, 5.7]) * 20).astype(int), cmap="Blues")

        # ax.hist2d(df.loc[:, x_dat[ii]], df.loc[:, y_dat[ii]], range=squarerange,
        #           bins=(np.array([8, 8]) * 10).astype(int), cmap="Blues")

        if x_weights is not None:
            x_scalar = x_weights[ii]
        else:
            x_scalar = 1

        xx = df.loc[:, x_dat[ii]] * x_scalar
        yy = df.loc[:, y_dat[ii]]

        # scatter plot
        ax[ii].scatter(xx, yy, alpha=.15, s=1)
        plt.ylim(lims)
        plt.xlim(lims)

        ax[ii].title.set_text(titles[ii])
        if ii == 0:
            ax[ii].set_ylabel(y_lab)
        if ii == 2:
            ax[ii].set_xlabel(x_lab)

        is_valid = ~np.isnan(xx) & ~np.isnan(yy)
        # cov.append(np.cov(xx[is_valid], yy[is_valid]))
        # rmsd.append(np.sqrt(np.nanmean((xx - yy) ** 2)))
        mb.append(np.nanmean(xx - yy))

    fig.add_subplot(111, frameon=False)
    if suptitle is not "":
        plt.suptitle(suptitle)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    # plt.xlabel("Lidar point reprojection")
    # plt.ylabel("Lidar ray sampling")

    return fig, ax, mb

## don't use contact number here!!!
# x_dat = ["contactnum_1",
#          "contactnum_2",
#          "contactnum_3",
#          "contactnum_4",
#          "contactnum_5"]

df_25.loc[:, "cn_from_tx_1"] = -np.log(df_25.loc[:, "transmission_s_1"])
df_25.loc[:, "cn_from_tx_2"] = -np.log(df_25.loc[:, "transmission_s_2"])
df_25.loc[:, "cn_from_tx_3"] = -np.log(df_25.loc[:, "transmission_s_3"])
df_25.loc[:, "cn_from_tx_4"] = -np.log(df_25.loc[:, "transmission_s_4"])
# df_25.loc[:, "cn_from_tx_5"] = -np.log(df_25.loc[:, "transmission_s_5"])

x_dat = ["cn_from_tx_1",
         "cn_from_tx_2",
         "cn_from_tx_3",
         "cn_from_tx_4"]

# y_dat = ["contactnum_1_pois",
#          "contactnum_2_pois",
#          "contactnum_3_pois",
#          "contactnum_4_pois",
#          "contactnum_5_pois"]

y_dat = ["lrs_cn_1",
         "lrs_cn_2",
         "lrs_cn_3",
         "lrs_cn_4"]

# y_dat = ["lrs_cn_1_snow_off_thresh",
#          "lrs_cn_2_snow_off_thresh",
#          "lrs_cn_3_snow_off_thresh",
#          "lrs_cn_4_snow_off_thresh"]


titles = ["0$^{\circ}$-15$^{\circ}$",
          "15$^{\circ}$-30$^{\circ}$",
          "30$^{\circ}$-45$^{\circ}$",
          "45$^{\circ}$-60$^{\circ}$"]

# x_weights = 1/np.cos((np.array([1, 2, 3, 4, 5]) * 15 - 15./2) * np.pi / 180)

# maxmin = [np.nanmin((df_25.loc[:, x_dat], df_25.loc[:, y_dat])) - .25,
#           np.nanmax((df_25.loc[:, x_dat], df_25.loc[:, y_dat])) + .25]
maxmin = [-0.2411994287105065, 7.549130176945413]

fig, ax, mb = plot_together(df_25, x_dat, y_dat, titles, lims=maxmin,
                        suptitle="Contact number comparison between methods over the forest plot",
                        y_lab=r"$\chi_{a-b}^{\blacktriangle}$ [-]",
                        # x_lab=r"$\chi_{a-b}^{\dagger}$ [-]")
                        x_lab=r"$\chi_{a-b}^{\bullet}$ [-]")
fig.savefig(plot_out_dir + "cn_comparison.png")
# fig.savefig(plot_out_dir + "cn_comparison_threshold.png")
# fig.savefig(plot_out_dir + "cn_comparison_threshold_lrs.png")

# aa = np.array([110, 100, 160, 160, 200, 200, 200, 200, 120, 120, 120, 120, 120, 130, 140])
# np.mean((aa / 255) ** (1/2.2))

x_dat = ["transmission_s_1",
         "transmission_s_2",
         "transmission_s_3",
         "transmission_s_4"]

y_dat = ["lrs_tx_1",
         "lrs_tx_2",
         "lrs_tx_3",
         "lrs_tx_4"]

# x_dat = ["lrs_tx_1_snow_off_thresh",
#          "lrs_tx_2_snow_off_thresh",
#          "lrs_tx_3_snow_off_thresh",
#          "lrs_tx_4_snow_off_thresh"]

titles = ["0$^{\circ}$-15$^{\circ}$",
          "15$^{\circ}$-30$^{\circ}$",
          "30$^{\circ}$-45$^{\circ}$",
          "45$^{\circ}$-60$^{\circ}$"]


fig, ax, mb = plot_together(df_25, x_dat, y_dat, titles, lims=[0, 1],
                        suptitle="Light transmittance comparison between methods over the forest plot",
                        y_lab=r"$T_{a-b}^{\blacktriangle}$ [-]",
                        # x_lab=r"$T_{a-b}^{\dagger}$ [-]")
                        x_lab=r"$T_{a-b}^{\bullet}$ [-]")
# fig.savefig(plot_out_dir + "tx_comparison.png")
# fig.savefig(plot_out_dir + "tx_comparison_threshold.png")
fig.savefig(plot_out_dir + "tx_comparison_threshold_lrs.png")

fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True, sharex=True, figsize=(8, 6), constrained_layout=True)
x_dat = ["lai_s_cc"]
# x_dat = ["lrs_tx_1"]
# x_dat = ["transmission_s_1"]
# x_dat = ["lrs_lai_15_deg_snow_on"]
# x_dat = ["dce"]
# x_dat = ["lrs_lai_60_deg"]
# x_dat = ["transmission_s_5"]
# y_dat = ["lai_s_cc_pois"]
# y_dat = ["lrs_lai_1_deg"]
y_dat = ["lrs_lai_2000"]
# y_dat = ['dswe_fnsd_19_050-19_052']
# y_dat = ['swe_fcon_19_050']
# y_dat = ["dce"]
# x_dat = ["lrs_lai_2000_snow_off_thresh"]
# y_dat = ["lrs_lai_60_deg"]
# x_dat = ["lrs_tx_5"]
# y_dat = ["lrs_cn_5"]
titles = ["LAI methods comparison over Upper Forest"]
# df = df_all
# ii = 0
x = [0, 100]
y = [0, 100]
plt.plot(x, y, color="black", linewidth=0.5)
# offset = 0
# plotrange = [[np.nanquantile(df.loc[:, x_dat], .0005) - offset, np.nanquantile(df.loc[:, x_dat], .995) + offset],
#                  [np.nanquantile(df.loc[:, y_dat], .0005) - offset, np.nanquantile(df.loc[:, y_dat], .995) + offset]]
# squarerange = [[np.min(plotrange), np.max(plotrange)],
#                [np.min(plotrange), np.max(plotrange)]]

# xx = -np.log(df.loc[:, x_dat])
# xx = df.loc[:, x_dat]
# yy = df.loc[:, y_dat]
xx = df_all.loc[:, x_dat]
yy = df_all.loc[:, y_dat]
maxmin = [np.nanmin((xx, yy)) - .25, np.nanmax((xx, yy)) + .25]
# maxmin = [1.9, 7.366133621664996]
# ax.hist2d(df.loc[:, x_dat[ii]], df.loc[:, y_dat[ii]], range=squarerange,
#           bins=(np.array([8, 8]) * 10).astype(int), cmap="Blues")
ax.scatter(xx, yy, alpha=.25, s=25)
plt.ylim(maxmin)
plt.xlim(maxmin)
ax.title.set_text(titles[ii])
ax.set_ylabel(r"$LAI_{2000}^{\blacktriangle}$ [-]")
# ax.set_xlabel(r"$LAI_{2000}^{\dagger}$ [-]")
ax.set_xlabel(r"$LAI_{2000}^{\bullet}$ [-]")
fig.savefig(plot_out_dir + "lai_comparison.png")
# fig.savefig(plot_out_dir + "lai_comparison_thresh.png")
# fig.savefig(plot_out_dir + "lai_comparison_thresh_lrs.png")

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


# stats
xx = "contactnum_5"
yy = "lrs_cn_5"

# xx = "transmission_s_5"
# yy = "lrs_tx_5"

# xx = "lai_s_cc"
# yy = "lrs_lai_2000"


valid = ~np.isnan(df_25[xx]) & ~np.isnan(df_25[yy])
pCor, pValue = pearsonr(df_25.loc[valid, xx], df_25.loc[valid, yy])
print("Pearsons: Correlaiton:{0} P-Value:{1}".format(pCor, pValue)) #print the P-Value and the T-Statistic
rmsd = np.sqrt(np.mean((df_25.loc[valid, yy] - df_25.loc[valid, xx]) ** 2))
print("RMSD: {0}".format(rmsd))
mb = np.mean(df_25.loc[valid, yy] - df_25.loc[valid, xx])
print("Mean Bias: {0}".format(mb))
rat = np.mean(df_25.loc[valid, xx] / df_25.loc[valid, yy])
print("Relative Ratio: {0}".format(rat))


# investigation into spatial differences

df_25_sub = df_25.loc[~np.isnan(df_25.transmission_s_1), :].copy()

df_25_sub.loc[:, "x_index_new"] = ((df_25_sub.x_index - np.nanmin(df_25_sub.x_index)) / 4).astype(int)
df_25_sub.loc[:, "y_index_new"] = ((df_25_sub.y_index - np.nanmin(df_25_sub.y_index)) / 4).astype(int)

xx = df_25_sub.lrs_cn_5
yy = -np.log(df_25_sub.transmission_s_5)

xx = df_25.lrs_cn_4
xx = df_25_sub.lrs_cn_4
yy = df_25_sub.contactnum_4

# xx = "lai_s_cc"
# yy = "lrs_lai_2000"

var_dif = yy - xx

map = np.full((np.nanmax(df_25_sub.y_index_new) + 1, np.nanmax(df_25_sub.x_index_new) + 1), np.nan)
map[df_25_sub.y_index_new, df_25_sub.x_index_new] = yy

map = np.full((np.nanmax(df_25.y_index) - np.nanmin(df_25.y_index) + 1, np.nanmax(df_25.x_index) - np.nanmin(df_25.x_index) + 1), np.nan)
map[df_25.y_index - np.nanmin(df_25.y_index), df_25.x_index - np.nanmin(df_25.x_index)] = xx

fig = plt.figure()
fig.subplots_adjust(top=0.90, bottom=0.15, left=0.15)
ax1 = fig.add_subplot(111)
# plt.scatter(x_coord[valid], y_coord[valid], c=var_dif[valid])
plt.imshow(map, interpolation="nearest", cmap="plasma", clim=(0, 4))
# plt.imshow(map, interpolation="nearest", cmap="RdYlBu", clim=(0, 1))
# plt.scatter(x_coord[valid], y_coord[valid], c=df_25.loc[valid, xx])
plt.colorbar()


### dce vs tx1

ddict = {
    'uf': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\site_library\\hemi_grid_points\\mb_65_r.25m_snow_off_offset0\\uf_plot_r.10m.tif',
    'lrs_id': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\site_library\\hemi_grid_points\\mb_65_r.25m_snow_off_offset.25\\dem_r.25_point_ids.tif',
}

merge_data_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_uf_r.25_px181_snow_off_dem_offset.25\\outputs\\rshmetalog_footprint_products.csv'

left_on = ['lrs_id']
right_on = ['id']
suffixes = ["_snow_off"]
data = merge_data_products(ddict, file_out, merge_data_in=merge_data_in, left_on=left_on, right_on=right_on, suffixes=suffixes, mode='median')

from libraries import raslib

ddict = {'uf': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\site_library\\hemi_grid_points\\mb_65_r.25m_snow_off_offset0\\uf_plot_r.10m.tif',
         'dce': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\DCE\\19_149_spike_free_chm_r.10m_dce.tif',
         'lrs_id': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\site_library\\hemi_grid_points\\mb_65_r.25m_snow_off_offset.25\\dem_r.25_point_ids.tif'
         }
df_oof = raslib.pd_sample_raster_gdal(ddict, include_nans=False, mode="nearest")
sup_data = pd.read_csv(merge_data_in)
df_fun = df_oof.merge(sup_data, left_on="lrs_id", right_on="id")

plt.scatter(df_fun.dce, df_fun.lrs_tx_1, alpha=.25)



