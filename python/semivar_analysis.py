import geotk
import numpy as np
import pandas as pd


# define sample density functions

def log10_inv(dd):
    return -np.log10(1 - dd)


def log_inv(dd):
    return -np.log(1 - dd)


def linear_ab(x):
    a = .1
    b = 10
    return (b - a) * x + a


def spatial_stats_on_col(df, colname, file_out=None, iterations=1000, replicates=1, nbins=50):


    # define points and values
    pts = df.loc[:, ['x_coord', 'y_coord']].values
    vals = df.loc[:, colname].values

    # sample point pairs
    df_samps, unif_bounds = geotk.pnt_sample_semivar(pts, vals, linear_ab, iterations, replicates, report_samp_vals=True)

    # compute stats on samples
    stats = geotk.bin_summarize(df_samps, linear_ab, unif_bounds, nbins, covar=True)

    if file_out is not None:
        # export to csv
        stats.to_csv(file_out, index=False)

    return stats

# load data
df_in ='C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\mb_15_merged_.25m_canopy_19_149.csv'
df = pd.read_csv(df_in)
df.loc[:, 'canopy_closure'] = 1 - df.openness

df.loc[:, 'cn_mean'] = df.loc[:, 'er_p0_mean'] * 0.19546
df.loc[:, 'cn_sd'] = df.loc[:, 'er_p0_sd'] * 0.19546

df.loc[:, ['cn_mean_ln']] = np.log(df.cn_mean)
df.loc[:, ['cn_mean_exp']] = np.exp(-df.cn_mean)
df.loc[:, ['cn_sd_ln']] = df.cn_sd * df.cn_mean_ln / df.cn_mean

# filter to uf site
df_uf = df[df.uf == 1]

stats_cn = spatial_stats_on_col(df_uf, 'cn_mean', iterations=10000, replicates=100, nbins=50)
stats_cn_log = spatial_stats_on_col(df_uf, 'cn_mean_ln', iterations=10000, replicates=100, nbins=50)
stats_dswe = spatial_stats_on_col(df_uf, 'dswe_19_045-19_052', iterations=10000, replicates=100, nbins=50)
stats_cn_exp = spatial_stats_on_col(df_uf, 'cn_mean_exp', iterations=10000, replicates=100, nbins=50)
stats_lai = spatial_stats_on_col(df_uf, 'lai_s_cc', iterations=10000, replicates=100, nbins=50)
stats_chm = spatial_stats_on_col(df_uf, 'chm', iterations=10000, replicates=100, nbins=50)
stats_dce = spatial_stats_on_col(df_uf, 'dce', iterations=10000, replicates=100, nbins=50)


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
stats = stats_chm
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# adding axes
axes.scatter(stats.mean_dist, stats.stdev)
axes.set_ylim([0, np.nanmax(stats.stdev) * 1.1])
axes.set_xlim([0, np.nanmax(stats.mean_dist) * 1.025])
plt.xlabel('Distance (m)')
plt.ylabel('Standard deviation of Contact Number')
plt.title('Standard deviation of contact number with distance\n Upper Forest, n=1000000')
plt.show()


# plt.scatter(stats.mean_dist, stats.mean_bias)
# plt.scatter(stats.bin_mid, stats.n)