import geotk
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

plot_out_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\graphics\\thesis_graphics\\semivar analysis\\"

# define sample density functions

def log10_inv(dd):
    return -np.log10(1 - dd)


def log_inv(dd):
    return -np.log(1 - dd)

max_dist = 25

def linear_ab(x):
    a = .05
    b = max_dist
    return (b - a) * x + a


def spatial_stats_on_col(df, colname, file_out=None, iterations=1000, replicates=1, nbins=50):

    # drop nan values
    valid = ~np.isnan(df.loc[:, colname].values)

    # define points and values
    pts = df.loc[valid, ['x_coord', 'y_coord']].values
    vals = df.loc[valid, colname].values

    # sample point pairs
    df_samps, unif_bounds = geotk.pnt_sample_semivar(pts, vals, linear_ab, iterations, replicates, report_samp_vals=True)
    # df_samps, unif_bounds = geotk.pnt_sample_semivar(pts, vals, log10_inv, iterations, replicates, report_samp_vals=True)

    # compute stats on samples
    stats = geotk.bin_summarize(df_samps, linear_ab, unif_bounds, nbins)

    if file_out is not None:
        # export to csv
        stats.to_csv(file_out, index=False)

    return stats, df_samps

# load data 5cm
data_05_in ='C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\uf_merged_.05m_ahpl_native.csv'
data_05 = pd.read_csv(data_05_in)
data_05_uf = data_05[data_05.uf == 1]

#load data 10cm
data_10_in ='C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\mb_15_merged_.10m_native_canopy_19_149.csv'
data_10 = pd.read_csv(data_10_in)
data_10_uf = data_10.loc[data_10.uf == 1, :]

# load data 25cm
data_25_in ='C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\mb_15_merged_.25m_native_canopy_19_149.csv'
data_25 = pd.read_csv(data_25_in)
data_25_uf = data_25.loc[data_25.uf == 1, :]
data_25_uf.loc[:, 'cn_mean'] = data_25_uf.er_p0_mean * 0.19447


#####
# sample

stats_swe_045, samps_045 = spatial_stats_on_col(data_05_uf, 'swe_19_045', iterations=10000, replicates=100, nbins=50)
stats_swe_045.to_csv(plot_out_dir + "semivar_stats_swe_045_md" + str(max_dist) + ".csv", index=False)
samps_045.to_csv(plot_out_dir + "semivar_samps_swe_045_md" + str(max_dist) + ".csv", index=False)

stats_swe_050, samps_050 = spatial_stats_on_col(data_05_uf, 'swe_19_050', iterations=10000, replicates=100, nbins=50)
stats_swe_050.to_csv(plot_out_dir + "semivar_stats_swe_050_md" + str(max_dist) + ".csv", index=False)
samps_050.to_csv(plot_out_dir + "semivar_samps_swe_050_md" + str(max_dist) + ".csv", index=False)

stats_swe_052, samps_052 = spatial_stats_on_col(data_05_uf, 'swe_19_052', iterations=10000, replicates=100, nbins=50)
stats_swe_052.to_csv(plot_out_dir + "semivar_stats_swe_052_md" + str(max_dist) + ".csv", index=False)
samps_052.to_csv(plot_out_dir + "semivar_samps_swe_052_md" + str(max_dist) + ".csv", index=False)

stats_dswe_045_050, samps_045_050 = spatial_stats_on_col(data_05_uf, 'dswe_19_045-19_050', iterations=10000, replicates=100, nbins=50)
stats_dswe_045_050.to_csv(plot_out_dir + "semivar_stats_dswe_045_050_md" + str(max_dist) + ".csv", index=False)
samps_045_050.to_csv(plot_out_dir + "semivar_samps_dswe_045_050_md" + str(max_dist) + ".csv", index=False)

stats_dswe_050_052, samps_050_052 = spatial_stats_on_col(data_05_uf, 'dswe_19_050-19_052', iterations=10000, replicates=100, nbins=50)
stats_dswe_050_052.to_csv(plot_out_dir + "semivar_stats_dswe_050_052_md" + str(max_dist) + ".csv", index=False)
samps_050_052.to_csv(plot_out_dir + "semivar_samps_dswe_050_052_md" + str(max_dist) + ".csv", index=False)

stats_chm, samps_chm = spatial_stats_on_col(data_10_uf, 'chm', iterations=10000, replicates=100, nbins=50)
stats_chm.to_csv(plot_out_dir + "semivar_stats_chm_md" + str(max_dist) + ".csv", index=False)
samps_chm.to_csv(plot_out_dir + "semivar_samps_chm_md" + str(max_dist) + ".csv", index=False)

stats_lai_rs, samps_lai_rs = spatial_stats_on_col(data_25_uf, 'cn_mean', iterations=10000, replicates=1000, nbins=50)
stats_lai_rs.to_csv(plot_out_dir + "semivar_stats_lai_rs_md" + str(max_dist) + ".csv", index=False)
samps_lai_rs.to_csv(plot_out_dir + "semivar_samps_lai_rs_md" + str(max_dist) + ".csv", index=False)

stats_lai_hemi, samps_lai_hemi = spatial_stats_on_col(data_25_uf, 'lai_s_cc', iterations=10000, replicates=1000, nbins=50)
stats_lai_hemi.to_csv(plot_out_dir + "semivar_stats_lai_hemi_md" + str(max_dist) + ".csv", index=False)
samps_lai_hemi.to_csv(plot_out_dir + "semivar_samps_lai_hemi_md" + str(max_dist) + ".csv", index=False)

# plot
stats = stats_swe_045
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# adding axes
# axes.scatter(stats.mean_dist, stats.stdev)
axes.plot(stats.mean_dist, stats.stdev)
axes.set_ylim([0, np.nanmax(stats.stdev) * 1.1])
axes.set_xlim([0, np.nanmax(stats.mean_dist) * 1.025])
plt.xlabel('Distance [m]')
plt.ylabel('Standard deviation of SWE [mm]')
plt.title('Standard deviation of SWE with distance\n 14 Feb. 2019, Upper Forest, 5cm resolution, n=1000000')
fig.savefig(plot_out_dir + "semivar_swe_045.png")


stats = stats_swe_050
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# adding axes
axes.plot(stats.mean_dist, stats.stdev)
axes.set_ylim([0, np.nanmax(stats.stdev) * 1.1])
axes.set_xlim([0, np.nanmax(stats.mean_dist) * 1.025])
plt.xlabel('Distance [m]')
plt.ylabel('Standard deviation of SWE [mm]')
plt.title('Standard deviation of SWE with distance\n 19 Feb. 2019, Upper Forest, 5cm resolution, n=1000000')
fig.savefig(plot_out_dir + "semivar_swe_050.png")



stats = stats_swe_052
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# adding axes
axes.plot(stats.mean_dist, stats.stdev)
axes.set_ylim([0, np.nanmax(stats.stdev) * 1.1])
axes.set_xlim([0, np.nanmax(stats.mean_dist) * 1.025])
plt.xlabel('Distance [m]')
plt.ylabel('Standard deviation of SWE [mm]')
plt.title('Standard deviation of SWE with distance\n 21 Feb. 2019, Upper Forest, 5cm resolution, n=1000000')
fig.savefig(plot_out_dir + "semivar_swe_052.png")




stats = stats_dswe_045_050
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# adding axes
axes.plot(stats.mean_dist, stats.stdev)
axes.set_ylim([0, np.nanmax(stats.stdev) * 1.1])
axes.set_xlim([0, np.nanmax(stats.mean_dist) * 1.025])
plt.xlabel('Distance [m]')
plt.ylabel('Standard deviation of $\Delta$SWE [mm]')
plt.title('Standard deviation of $\Delta$SWE with distance\n 14-19 Feb. 2019, Upper Forest, 5cm resolution, n=1000000')
fig.savefig(plot_out_dir + "semivar_dswe_045_050.png")




stats = stats_dswe_050_052
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# adding axes
axes.plot(stats.mean_dist, stats.stdev)
axes.set_ylim([0, np.nanmax(stats.stdev) * 1.1])
axes.set_xlim([0, np.nanmax(stats.mean_dist) * 1.025])
plt.xlabel('Distance [m]')
plt.ylabel('Standard deviation of $\Delta$SWE [mm]')
plt.title('Standard deviation of $\Delta$SWE with distance\n 19-21 Feb. 2019, Upper Forest, 5cm resolution, n=1000000')
fig.savefig(plot_out_dir + "semivar_dswe_050_052.png")




stats = stats_chm
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# adding axes
axes.plot(stats.mean_dist, stats.stdev)
axes.set_ylim([0, np.nanmax(stats.stdev) * 1.1])
axes.set_xlim([0, np.nanmax(stats.mean_dist) * 1.025])
plt.xlabel('Distance [m]')
plt.ylabel('Standard deviation of canopy height [m]')
plt.title('Standard deviation of canopy height with distance\n Upper Forest, 10m resolution, n=1000000')
fig.savefig(plot_out_dir + "semivar_chm.png")




stats = stats_lai_rs
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# adding axes
axes.plot(stats.mean_dist, stats.stdev)
axes.set_ylim([0, np.nanmax(stats.stdev) * 1.1])
axes.set_xlim([0, np.nanmax(stats.mean_dist) * 1.025])
plt.xlabel('Distance [m]')
plt.ylabel('Standard deviation of Ray Sampled LAI [-]')
plt.title('Standard deviation of Ray Sampled LAI with distance\n Upper Forest, 5cm resolution, n=10000000')
fig.savefig(plot_out_dir + "semivar_lai_rs.png")




stats = stats_lai_hemi
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# adding axes
axes.plot(stats.mean_dist, stats.stdev)
axes.set_ylim([0, np.nanmax(stats.stdev) * 1.1])
axes.set_xlim([0, np.nanmax(stats.mean_dist) * 1.025])
plt.xlabel('Distance [m]')
plt.ylabel('Standard deviation of Hemi-photo LAI [-]')
plt.title('Standard deviation of Hemi-photo LAI with distance\n Upper Forest, 25cm resolution, n=10000000')
fig.savefig(plot_out_dir + "semivar_lai_hemi.png")


# # combined plot
# samps_045 = pd.read_csv(plot_out_dir + "semivar_samps_swe_045_md" + str(max_dist) + ".csv")
# samps_050 = pd.read_csv(plot_out_dir + "semivar_samps_swe_050_md" + str(max_dist) + ".csv")
# samps_052 = pd.read_csv(plot_out_dir + "semivar_samps_swe_052_md" + str(max_dist) + ".csv")
# samps_045_050 = pd.read_csv(plot_out_dir + "semivar_samps_dswe_045_050_md" + str(max_dist) + ".csv")
# samps_050_052 = pd.read_csv(plot_out_dir + "semivar_samps_dswe_050_052_md" + str(max_dist) + ".csv")
# samps_chm = pd.read_csv(plot_out_dir + "semivar_samps_chm_md" + str(max_dist) + ".csv")
# samps_lai_rs = pd.read_csv(plot_out_dir + "semivar_samps_lai_rs_md" + str(max_dist) + ".csv")
# samps_lai_hemi = pd.read_csv(plot_out_dir + "semivar_samps_lai_hemi_md" + str(max_dist) + ".csv")

# stats_045 = pd.read_csv(plot_out_dir + "semivar_stats_swe_045_md" + str(max_dist) + ".csv")
# stats_050 = pd.read_csv(plot_out_dir + "semivar_stats_swe_050_md" + str(max_dist) + ".csv")
# stats_052 = pd.read_csv(plot_out_dir + "semivar_stats_swe_052_md" + str(max_dist) + ".csv")
# stats_045_050 = pd.read_csv(plot_out_dir + "semivar_stats_dswe_045_050_md" + str(max_dist) + ".csv")
# stats_050_052 = pd.read_csv(plot_out_dir + "semivar_stats_dswe_050_052_md" + str(max_dist) + ".csv")
# stats_chm = pd.read_csv(plot_out_dir + "semivar_stats_chm_md" + str(max_dist) + ".csv")
# stats_lai_rs = pd.read_csv(plot_out_dir + "semivar_stats_lai_rs_md" + str(max_dist) + ".csv")
# stats_lai_hemi = pd.read_csv(plot_out_dir + "semivar_stats_lai_hemi_md" + str(max_dist) + ".csv")

sd_045 = np.nanstd(data_05_uf.swe_19_045)
sd_050 = np.nanstd(data_05_uf.swe_19_050)
sd_052 = np.nanstd(data_05_uf.swe_19_052)
sd_045_050 = np.nanstd(data_05_uf.loc[:, 'dswe_19_045-19_050'])
sd_050_052 = np.nanstd(data_05_uf.loc[:, 'dswe_19_050-19_052'])
sd_chm = np.nanstd(data_10_uf.chm)
sd_lai_rs = np.nanstd(data_25_uf.cn_mean)
sd_lai_hemi = np.nanstd(data_25_uf.lai_s_cc)

# 2.5m
n_bins = 50
d_bounds = (0, 2.5)
stats_045 = geotk.bin_summarize(samps_045, n_bins, d_bounds=d_bounds)
stats_050 = geotk.bin_summarize(samps_050, n_bins, d_bounds=d_bounds)
stats_052 = geotk.bin_summarize(samps_052, n_bins, d_bounds=d_bounds)
stats_045_050 = geotk.bin_summarize(samps_045_050, n_bins, d_bounds=d_bounds)
stats_050_052 = geotk.bin_summarize(samps_050_052, n_bins, d_bounds=d_bounds)
stats_chm = geotk.bin_summarize(samps_chm, n_bins, d_bounds=d_bounds)
stats_lai_rs = geotk.bin_summarize(samps_lai_rs, n_bins, d_bounds=d_bounds)
stats_lai_hemi = geotk.bin_summarize(samps_lai_hemi, n_bins, d_bounds=d_bounds)


fig = plt.figure()
axes = fig.add_axes([0.13, 0.1, 0.62, 0.8])
# adding axes
plt.xlabel('Distance (d) [m]')
plt.ylabel('Standard semi-variance $\left( \\frac{var(x_{d})}{2 \cdot var(x)}\\right)$ [-]')
plt.title('Standard semi-variance of metrics with distance\n Upper Forest, 5-25cm resolution, n=100000')

axes.plot(stats_045.mean_dist, .5 * stats_045.stdev**2 / sd_045**2, label="SWE 045", linestyle="dashed")
axes.plot(stats_050.mean_dist, .5 * stats_050.stdev**2 / sd_050**2, label="SWE 050", linestyle="dashed")
axes.plot(stats_052.mean_dist, .5 * stats_052.stdev**2 / sd_052**2, label="SWE 052", linestyle="dashed")

axes.plot(stats_045_050.mean_dist, .5 * stats_045_050.stdev**2 / sd_045_050**2, label="$\Delta$SWE 045-050", linestyle="dotted")
axes.plot(stats_050_052.mean_dist, .5 * stats_050_052.stdev**2 / sd_050_052**2, label="$\Delta$SWE 050-052", linestyle="dotted")

axes.plot(stats_chm.mean_dist, .5 * stats_chm.stdev**2 / sd_chm**2, label="Canopy height")
axes.plot(stats_lai_rs.mean_dist, .5 * stats_lai_rs.stdev**2 / sd_lai_rs**2, label="$LAI_{rs}$")
axes.plot(stats_lai_hemi.mean_dist, .5 * stats_lai_hemi.stdev**2 / sd_lai_hemi**2, label="$LAI_{hemi}$")

axes.legend(loc='center left', bbox_to_anchor=(1.01, .5), ncol=1,
            borderaxespad=0, frameon=False)
fig.savefig(plot_out_dir + "semivar_combined_2.5m.png")


# 25m
n_bins = 50
d_bounds = (0, 25)
stats_045 = geotk.bin_summarize(samps_045, n_bins, d_bounds=d_bounds)
stats_050 = geotk.bin_summarize(samps_050, n_bins, d_bounds=d_bounds)
stats_052 = geotk.bin_summarize(samps_052, n_bins, d_bounds=d_bounds)
stats_045_050 = geotk.bin_summarize(samps_045_050, n_bins, d_bounds=d_bounds)
stats_050_052 = geotk.bin_summarize(samps_050_052, n_bins, d_bounds=d_bounds)
stats_chm = geotk.bin_summarize(samps_chm, n_bins, d_bounds=d_bounds)
stats_lai_rs = geotk.bin_summarize(samps_lai_rs, n_bins, d_bounds=d_bounds)
stats_lai_hemi = geotk.bin_summarize(samps_lai_hemi, n_bins, d_bounds=d_bounds)


fig = plt.figure()
axes = fig.add_axes([0.13, 0.1, 0.62, 0.8])
# adding axes
plt.xlabel('Distance (d) [m]')
plt.ylabel('Standard semi-variance $\left( \\frac{var(x_{d})}{2 \cdot var(x)}\\right)$ [-]')
plt.title('Standard semi-variance of metrics with distance\n Upper Forest, 5-25cm resolution, n=1000000')

axes.plot(stats_045.mean_dist, .5 * stats_045.stdev**2 / sd_045**2, label="SWE 045", linestyle="dashed")
axes.plot(stats_050.mean_dist, .5 * stats_050.stdev**2 / sd_050**2, label="SWE 050", linestyle="dashed")
axes.plot(stats_052.mean_dist, .5 * stats_052.stdev**2 / sd_052**2, label="SWE 052", linestyle="dashed")

axes.plot(stats_045_050.mean_dist, .5 * stats_045_050.stdev**2 / sd_045_050**2, label="$\Delta$SWE 045-050", linestyle="dotted")
axes.plot(stats_050_052.mean_dist, .5 * stats_050_052.stdev**2 / sd_050_052**2, label="$\Delta$SWE 050-052", linestyle="dotted")

axes.plot(stats_chm.mean_dist, .5 * stats_chm.stdev**2 / sd_chm**2, label="Canopy height")
axes.plot(stats_lai_rs.mean_dist, .5 * stats_lai_rs.stdev**2 / sd_lai_rs**2, label="$LAI_{rs}$")
axes.plot(stats_lai_hemi.mean_dist, .5 * stats_lai_hemi.stdev**2 / sd_lai_hemi**2, label="$LAI_{hemi}$")

axes.legend(loc='center left', bbox_to_anchor=(1.01, .5), ncol=1,
            borderaxespad=0, frameon=False)
fig.savefig(plot_out_dir + "semivar_combined_25m.png")