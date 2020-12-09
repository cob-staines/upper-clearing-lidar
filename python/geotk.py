# # toolkit for geographical analysis
#
# # histograms
# import laslib
# import numpy as np
# import pandas as pd
#
# def pnt_sample_semivar(pts_1, vals_1, dist_inv_func, n_samples, n_iters=1, pts_2=None, vals_2=None, self_ref=False):
#     if pts_2 is None:
#         pts_2 = pts_1
#
#     if vals_2 is None:
#         vals_2 = vals_1
#
#     # select a set of points at random
#     samps = np.random.randint(0, high=pts_1.__len__(), size=n_samples)
#
#     # sample target distances according to distribution
#     unif_samps = np.random.random((n_samples, n_iters))
#     targ_dist = dist_inv_func(unif_samps)
#
#     # preallocate distance and difference vectors
#     true_dist = np.full((n_samples, n_iters), np.nan)
#     dv = np.full((n_samples, n_iters), np.nan)
#
#     for ii in range(0, n_samples):
#         # for each sample calculate distances to all points
#         dd = np.sqrt(np.sum((pts_2 - pts_1[samps[ii], :]) ** 2, axis=1))
#         # dd = np.sqrt((pts_2.x - pts_1.x[samps[ii]]) ** 2 + (pts_2.y - pts_1.y[samps[ii]]) ** 2)
#         for jj in range(0, n_iters):
#             # for each iteration, find the point with distance closest to the corresponding target distance
#             idx = (np.abs(dd - targ_dist[ii, jj])).argmin()
#             # record the actual distance between points
#             true_dist[ii, jj] = dd[idx]
#             # record value difference
#             dv[ii, jj] = vals_2[idx] - vals_2[samps[ii]]
#         print(ii + 1)
#
#     df = pd.DataFrame({'dist': true_dist.reshape(n_samples * n_iters),
#                        'dvals': dv.reshape(n_samples * n_iters)})
#     if not self_ref:
#         # remove self-referencing values
#         df = df[df.dist != 0]
#
#     unif_bounds = (np.min(unif_samps), np.max(unif_samps))  # not perfect, as target distances do not necesarily equal true distances.
#
#     return df, unif_bounds
#
# def bin_summarize(df, dist_inv_func, bounds, bin_count):
#     dd = df.dist
#     vv = df.dvals
#
#     # calculate bins according to dist_inv_func
#     bins = dist_inv_func(np.linspace(bounds[0], bounds[1], bin_count + 1))
#
#     bin_mid = (bins[0:-1] + bins[1:]) / 2
#     # bin df according to a
#     v_groups = vv.groupby(np.digitize(dd, bins[0:-1]))
#     d_groups = dd.groupby(np.digitize(dd, bins[0:-1]))
#     stats = pd.DataFrame({'bin_low': bins[0:-1],
#                           'bin_mid': bin_mid,
#                           'bin_high': bins[1:],
#                           'n': 0,
#                           'mean_dist': np.nan,
#                           'mean_bias': np.nan,
#                           'stdev': np.nan})
#
#     # report groups with counts of 0
#     non_empty = np.array(v_groups.count().index) - 1
#     stats.loc[non_empty, 'n'] = np.array(v_groups.count())
#     stats.loc[non_empty, 'mean_dist'] = np.array(d_groups.mean())
#     stats.loc[non_empty, 'mean_bias'] = np.array(v_groups.mean())
#     stats.loc[non_empty, 'stdev'] = np.array(v_groups.std())
#
#     return stats
#
# # semivar (self referencing error with distance)
#
# # # las example
# # las_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\LAS\\19_149_UF.las'
# # pts_xyz = laslib.las_xyz_load(las_in, keep_class=2)
# # pts = pts_xyz[:, [0, 1]]
# # vals = pts_xyz[:, 2]
# # # pts = pd.DataFrame(data=pts, columns=['x', 'y', 'z'])
# #
# # df, unif_bounds = geotk.pnt_sample_semivar(pts, vals, linear_ab, 10, 100)
# # stats = geotk.bin_summarize(df, linear_ab, unif_bounds, 25)
#
# # plot
# import pandas as pd
# import numpy as np
# data_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\mb_15_merged_.10m_canopy_19_149.csv'
# data = pd.read_csv(data_in)
#
# # counts, bins = np.histogram(data.er_p0_mean[~np.isnan(data.er_p0_mean)], bins=50)
# # norm = np.sum(counts)
# #
# # lai_stats = pd.DataFrame({'bin_low': bins[0:-1],
# #                           'bin_mid': (bins[0:-1] + bins[1:]) / 2,
# #                           'bin_high': bins[1:],
# #                           'density': counts / norm,
# #                           'log_density': np.log(counts / norm)})
# #
# # counts, bins = np.histogram(data.er_p0_mean[~np.isnan(data.loc[:, 'dswe_19_045-19_052'])], bins=bins)
# # swe_stats = pd.DataFrame({'bin_low': bins[0:-1],
# #                           'bin_mid': (bins[0:-1] + bins[1:]) / 2,
# #                           'bin_high': bins[1:],
# #                           'density': counts / norm,
# #                           'log_density': np.log(counts / norm)})
#
# #### rejection sample points
# proposal = 'dswe_19_045-19_052'  # given these values
# sample = 'er_p0_mean'  # sample according to this distribution
#
# nbins = 25
# nsamps = 100000
# # data
#
# valid_samp = ~np.isnan(data.loc[:, sample])
# valid_prop = ~np.isnan(data.loc[:, proposal])
#
# # determine bins by equal quantiles of sample data
# scrap, bins = pd.qcut(data.loc[~np.isnan(data.loc[:, sample]), sample], q=nbins, retbins=True)
# # # determine bins by equal interval of sample data
# # scrap, bins = np.histogram(data.loc[valid_samp, sample], bins=nbins)
#
# # hist of native sample dist
# samp_count, bins = np.histogram(data.loc[valid_samp, sample], bins=bins)
#
# # histogram of observed (valid) sample distribution
# prop_count, bins = np.histogram(data.loc[valid_prop, sample], bins=bins)
#
# stats = pd.DataFrame({'bin_low': bins[0:-1],
#                       'bin_mid': (bins[0:-1] + bins[1:]) / 2,
#                       'bin_high': bins[1:],
#                       'prop_dist': prop_count/np.sum(prop_count),
#                       'samp_dist': samp_count/np.sum(samp_count)})
#
# stats = stats.assign(dist_ratio=stats.prop_dist / stats.samp_dist)
# stats = stats.assign(samp_scaled=stats.samp_dist * np.min(stats.dist_ratio))
# stats = stats.assign(acc_rate=stats.samp_scaled / stats.prop_dist)
#
# total_acc_rate = np.sum(prop_count * stats.acc_rate) / np.sum(prop_count)
#
# # randomly sample from valid_prop
# # prop_id = np.random.randint(0, len(data), nsamps)  # with replacement
# prop_samps = np.int(nsamps / total_acc_rate) + 1
#
# prop_id = np.random.permutation(range(0, len(data.loc[valid_prop, :])))[0:prop_samps]  # without replacement (nsamps must be less than data length)
# d_prop = data.loc[valid_prop, :].iloc[prop_id, :]
#
# rs = pd.DataFrame({'cat': np.digitize(d_prop.loc[:, sample], bins=bins) - 1,
#                    'seed': np.random.random(len(d_prop))},
#                    index=d_prop.index)
#
# rs = pd.merge(rs, stats.acc_rate, how='left', left_on='cat', right_index=True)
#
# accept = (rs.seed <= rs.acc_rate)
#
# d_samp = d_prop.loc[accept, :]
#
# # evaluate final dist
#
# scrap, bins = np.histogram(data.loc[valid_samp, sample], bins=nbins)
# samp_count, bins = np.histogram(data.loc[valid_samp, sample], bins=bins)
# prop_count, bins = np.histogram(data.loc[valid_prop, sample], bins=bins)
# d_samp_count, bins = np.histogram(d_samp.loc[:, sample], bins=bins)
#
# eval = pd.DataFrame({'bin_low': bins[0:-1],
#                       'bin_mid': (bins[0:-1] + bins[1:]) / 2,
#                       'bin_high': bins[1:],
#                       'prop_dist': prop_count/np.sum(prop_count),
#                       'samp_dist': samp_count/np.sum(samp_count),
#                       'd_samp_dist': d_samp_count/np.sum(d_samp_count)})
# eval = eval.assign(prop_ratio=eval.prop_dist / eval.prop_dist)
# eval = eval.assign(samp_ratio=eval.prop_dist / eval.samp_dist)
# eval = eval.assign(d_samp_ratio=eval.prop_dist / eval.d_samp_dist)
#
#
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
#
# plt.plot(stats.bin_mid, stats.prop_dist)
# plt.plot(stats.bin_mid, stats.samp_dist)
# plt.plot(stats.bin_mid, stats.d_samp_dist)
# plt.plot(stats.bin_mid, stats.dist_ratio)
# plt.plot(stats.bin_mid, stats.samp_scaled)
# plt.plot(stats.bin_mid, stats.acc_rate)
#
# plt.plot(eval.bin_mid, eval.prop_ratio)
# plt.plot(eval.bin_mid, eval.samp_ratio)
# plt.plot(eval.bin_mid, eval.d_samp_ratio)
#
# plt.plot(eval.bin_mid, eval.prop_dist)
# plt.plot(eval.bin_mid, eval.samp_dist)
# plt.plot(eval.bin_mid, eval.d_samp_dist)


import time
from tqdm import tqdm

for i in tqdm(range(5), position=1, desc="i", leave=False, ncols=80):
    for j in tqdm(range(10), position=2, desc="j", leave=False, ncols=80):
        time.sleep(0.5)

