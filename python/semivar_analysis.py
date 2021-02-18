import geotk
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plot_out_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\graphics\\thesis_graphics\\semivar analysis\\"

# define sample density functions

def log10_inv(dd):
    return -np.log10(1 - dd)


def log_inv(dd):
    return -np.log(1 - dd)


def linear_ab(x):
    a = .05
    b = 10
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
data_05_in ='C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\uf_merged_ajli_.05m_canopy_19_149.csv'
data_05 = pd.read_csv(data_05_in)

#load data 10cm
data_10_in ='C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\mb_15_merged_.10m_canopy_19_149.csv'
data_10 = pd.read_csv(data_10_in)
data_10_uf = data_10[data_10.uf == 1]

# load data 25cm
data_25_in ='C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\mb_15_merged_.25m_canopy_19_149.csv'
data_25 = pd.read_csv(data_25_in)
data_25_uf = data_25[data_25.uf == 1]
data_25.loc[:, 'cn_mean'] = data_25.loc[:, 'er_p0_mean'] * 0.19447


#####
# sample and plot

stats_swe_045, samps_045 = spatial_stats_on_col(data_05, 'swe_19_045_1', iterations=10000, replicates=100, nbins=50)
stats = stats_swe_045
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# adding axes
axes.scatter(stats.mean_dist, stats.stdev)
axes.set_ylim([0, np.nanmax(stats.stdev) * 1.1])
axes.set_xlim([0, np.nanmax(stats.mean_dist) * 1.025])
plt.xlabel('Distance [m]')
plt.ylabel('Standard deviation of SWE [mm]')
plt.title('Standard deviation of SWE with distance\n 14 Feb. 2019, Upper Forest, 5cm resolution, n=1000000')
fig.savefig(plot_out_dir + "semivar_swe_045.png")


stats_swe_050, samps_050 = spatial_stats_on_col(data_05, 'swe_19_050_2', iterations=10000, replicates=100, nbins=50)
stats = stats_swe_050
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# adding axes
axes.scatter(stats.mean_dist, stats.stdev)
axes.set_ylim([0, np.nanmax(stats.stdev) * 1.1])
axes.set_xlim([0, np.nanmax(stats.mean_dist) * 1.025])
plt.xlabel('Distance [m]')
plt.ylabel('Standard deviation of SWE [mm]')
plt.title('Standard deviation of SWE with distance\n 19 Feb. 2019, Upper Forest, 5cm resolution, n=1000000')
fig.savefig(plot_out_dir + "semivar_swe_050.png")


stats_swe_052, samps_052 = spatial_stats_on_col(data_05, 'swe_19_052_2', iterations=10000, replicates=100, nbins=50)
stats = stats_swe_052
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# adding axes
axes.scatter(stats.mean_dist, stats.stdev)
axes.set_ylim([0, np.nanmax(stats.stdev) * 1.1])
axes.set_xlim([0, np.nanmax(stats.mean_dist) * 1.025])
plt.xlabel('Distance [m]')
plt.ylabel('Standard deviation of SWE [mm]')
plt.title('Standard deviation of SWE with distance\n 21 Feb. 2019, Upper Forest, 5cm resolution, n=1000000')
fig.savefig(plot_out_dir + "semivar_swe_052.png")


stats_dswe_045_050, samps_045_050 = spatial_stats_on_col(data_05, 'dswe_19_045-19_050', iterations=10000, replicates=100, nbins=50)
stats = stats_dswe_045_050
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# adding axes
axes.scatter(stats.mean_dist, stats.stdev)
axes.set_ylim([0, np.nanmax(stats.stdev) * 1.1])
axes.set_xlim([0, np.nanmax(stats.mean_dist) * 1.025])
plt.xlabel('Distance [m]')
plt.ylabel('Standard deviation of $\Delta$SWE [mm]')
plt.title('Standard deviation of $\Delta$SWE with distance\n 14-19 Feb. 2019, Upper Forest, 5cm resolution, n=1000000')
fig.savefig(plot_out_dir + "semivar_dswe_045_050.png")


stats_dswe_050_052, samps_050_052 = spatial_stats_on_col(data_05, 'dswe_19_050-19_052', iterations=10000, replicates=100, nbins=50)
stats = stats_dswe_050_052
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# adding axes
axes.scatter(stats.mean_dist, stats.stdev)
axes.set_ylim([0, np.nanmax(stats.stdev) * 1.1])
axes.set_xlim([0, np.nanmax(stats.mean_dist) * 1.025])
plt.xlabel('Distance [m]')
plt.ylabel('Standard deviation of $\Delta$SWE [mm]')
plt.title('Standard deviation of $\Delta$SWE with distance\n 19-21 Feb. 2019, Upper Forest, 5cm resolution, n=1000000')
fig.savefig(plot_out_dir + "semivar_dswe_050_052.png")


stats_chm, samps_chm = spatial_stats_on_col(data_10, 'chm', iterations=10, replicates=100, nbins=50)
stats = stats_chm
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# adding axes
axes.scatter(stats.mean_dist, stats.stdev)
axes.set_ylim([0, np.nanmax(stats.stdev) * 1.1])
axes.set_xlim([0, np.nanmax(stats.mean_dist) * 1.025])
plt.xlabel('Distance [m]')
plt.ylabel('Standard deviation of canopy height [m]')
plt.title('Standard deviation of canopy height with distance\n Upper Forest, 10m resolution, n=1000000')
fig.savefig(plot_out_dir + "semivar_chm.png")

stats_lai_rs, samps_lai_rs = spatial_stats_on_col(data_25, 'cn_mean', iterations=10000, replicates=100, nbins=50)
stats = stats_lai_rs
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# adding axes
axes.scatter(stats.mean_dist, stats.stdev)
axes.set_ylim([0, np.nanmax(stats.stdev) * 1.1])
axes.set_xlim([0, np.nanmax(stats.mean_dist) * 1.025])
plt.xlabel('Distance [m]')
plt.ylabel('Standard deviation of Ray Sampled LAI [-]')
plt.title('Standard deviation of Ray Sampled LAI with distance\n Upper Forest, 5cm resolution, n=1000000')
fig.savefig(plot_out_dir + "semivar_lai_rs.png")

stats_lai_hemi, samps_lai_hemi = spatial_stats_on_col(data_25, 'lai_s_cc', iterations=10000, replicates=100, nbins=50)
stats = stats_lai_hemi
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# adding axes
axes.scatter(stats.mean_dist, stats.stdev)
axes.set_ylim([0, np.nanmax(stats.stdev) * 1.1])
axes.set_xlim([0, np.nanmax(stats.mean_dist) * 1.025])
plt.xlabel('Distance [m]')
plt.ylabel('Standard deviation of Hemi-photo LAI [-]')
plt.title('Standard deviation of Hemi-photo LAI with distance\n Upper Forest, 25cm resolution, n=1000000')
fig.savefig(plot_out_dir + "semivar_lai_hemi.png")