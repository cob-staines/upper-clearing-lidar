import geotk
import numpy as np
import pandas as pd

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

# load data
data_in ='C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\uf_merged_ajli_.05m_canopy_19_149.csv'
data = pd.read_csv(data_in)

stats_dswe_045_050, samps_045_050 = spatial_stats_on_col(data, 'dswe_19_045-19_050', iterations=10000, replicates=100, nbins=50)
stats_dswe_050_052, samps_050_052 = spatial_stats_on_col(data, 'dswe_19_050-19_052', iterations=10000, replicates=100, nbins=50)

stats_swe_045, samps_045 = spatial_stats_on_col(data, 'swe_19_045_1', iterations=10000, replicates=100, nbins=50)
stats_swe_050, samps_050 = spatial_stats_on_col(data, 'swe_19_050_2', iterations=10000, replicates=100, nbins=50)
stats_swe_052, samps_052 = spatial_stats_on_col(data, 'swe_19_052_2', iterations=10000, replicates=100, nbins=50)

# load data
data_2_in ='C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\mb_15_merged_.25m_canopy_19_149.csv'
data_2 = pd.read_csv(data_2_in)
data_2_uf = data_2[data_2.uf == 1]
data_2.loc[:, 'cn_mean'] = data_2.loc[:, 'er_p0_mean'] * 0.19447

# LAI_rs
stats_lai_hemi, samps_lai_hemi = spatial_stats_on_col(data_2, 'lai_s_cc', iterations=10000, replicates=100, nbins=50)
stats_cn, samps_cn = spatial_stats_on_col(data_2, 'cn_mean', iterations=10000, replicates=100, nbins=50)


#####
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# plt.scatter(stats.mean_dist, stats.stdev)
# plt.ylim(0, np.max(stats.stdev))
# plt.xlim(0, np.max(stats.mean_dist))
# # plt.show()
#
# fig, ax = plt.subplots()
stats = stats_cn
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# adding axes
axes.scatter(stats.mean_dist, stats.stdev)
axes.set_ylim([0, np.nanmax(stats.stdev) * 1.1])
axes.set_xlim([0, np.nanmax(stats.mean_dist) * 1.025])
plt.xlabel('Distance (m)')
plt.ylabel('Standard deviation of Contact Number')
plt.title('Standard deviation of Ray Sampled LAI with distance\n Upper Forest, 5cm resolution, n=1000000')
fig.savefig(plot_out_dir + "semivar_lai_rs.png")


stats = stats_dswe_045_050
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# adding axes
axes.scatter(stats.mean_dist, stats.stdev)
axes.set_ylim([0, np.nanmax(stats.stdev) * 1.1])
axes.set_xlim([0, np.nanmax(stats.mean_dist) * 1.025])
plt.xlabel('Distance (m)')
plt.ylabel('Standard deviation of $\Delta$SWE')
plt.title('Standard deviation of $\Delta$SWE with distance\n 14-19 Feb. 2019, Upper Forest, 5cm resolution, n=1000000')
fig.savefig(plot_out_dir + "semivar_dswe_045_050.png")


stats = stats_dswe_050_052
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# adding axes
axes.scatter(stats.mean_dist, stats.stdev)
axes.set_ylim([0, np.nanmax(stats.stdev) * 1.1])
axes.set_xlim([0, np.nanmax(stats.mean_dist) * 1.025])
plt.xlabel('Distance (m)')
plt.ylabel('Standard deviation of $\Delta$SWE')
plt.title('Standard deviation of $\Delta$SWE with distance\n 19-21 Feb. 2019, Upper Forest, 5cm resolution, n=1000000')
fig.savefig(plot_out_dir + "semivar_dswe_050_052.png")


stats = stats_swe_045
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# adding axes
axes.scatter(stats.mean_dist, stats.stdev)
axes.set_ylim([0, np.nanmax(stats.stdev) * 1.1])
axes.set_xlim([0, np.nanmax(stats.mean_dist) * 1.025])
plt.xlabel('Distance (m)')
plt.ylabel('Standard deviation of SWE')
plt.title('Standard deviation of SWE with distance\n 14 Feb. 2019, Upper Forest, 5cm resolution, n=1000000')
fig.savefig(plot_out_dir + "semivar_swe_045.png")


stats = stats_swe_050
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# adding axes
axes.scatter(stats.mean_dist, stats.stdev)
axes.set_ylim([0, np.nanmax(stats.stdev) * 1.1])
axes.set_xlim([0, np.nanmax(stats.mean_dist) * 1.025])
plt.xlabel('Distance (m)')
plt.ylabel('Standard deviation of SWE')
plt.title('Standard deviation of SWE with distance\n 19 Feb. 2019, Upper Forest, 5cm resolution, n=1000000')
fig.savefig(plot_out_dir + "semivar_swe_050.png")


stats = stats_swe_052
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# adding axes
axes.scatter(stats.mean_dist, stats.stdev)
axes.set_ylim([0, np.nanmax(stats.stdev) * 1.1])
axes.set_xlim([0, np.nanmax(stats.mean_dist) * 1.025])
plt.xlabel('Distance (m)')
plt.ylabel('Standard deviation of SWE')
plt.title('Standard deviation of SWE with distance\n 21 Feb. 2019, Upper Forest, 5cm resolution, n=1000000')
fig.savefig(plot_out_dir + "semivar_swe_052.png")



stats = stats_lai_hemi
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# adding axes
axes.scatter(stats.mean_dist, stats.stdev)
axes.set_ylim([0, np.nanmax(stats.stdev) * 1.1])
axes.set_xlim([0, np.nanmax(stats.mean_dist) * 1.025])
plt.xlabel('Distance (m)')
plt.ylabel('Standard deviation of Contact Number')
plt.title('Standard deviation of Hemi-photo LAI with distance\n Upper Forest, 25cm resolution, n=1000000')
fig.savefig(plot_out_dir + "semivar_lai_hemi.png")