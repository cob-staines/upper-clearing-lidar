import geotk
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

plot_out_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\graphics\\thesis_graphics\\semivar analysis\\"

def spatial_stats_on_col(df, colname, dist_bounds, file_out=None, iterations=1000, replicates=1, nbins=50):

    # drop nan values
    valid = ~np.isnan(df.loc[:, colname].values)

    # define points and values
    pts = df.loc[valid, ['x_coord', 'y_coord']].values
    vals = df.loc[valid, colname].values

    # define inverse sample density functions
    # liear a to b
    def samp_dens_inv(x):
        a = dist_bounds[0]
        b = dist_bounds[1]
        return (b - a) * x + a

    # log
    # def samp_dens_inv(dd):
    #     return -np.log(1 - dd)

    # log10
    # def samp_dens_inv(dd):
    #     return -np.log10(1 - dd)


    # sample point pairs
    df_samps, unif_bounds = geotk.pnt_sample_semivar(pts, vals, samp_dens_inv, iterations, replicates, report_samp_vals=True)

    # compute stats on samples
    stats = geotk.bin_summarize(df_samps, nbins, dist_inv_func=samp_dens_inv, unif_bounds=unif_bounds)

    if file_out is not None:
        # export to csv
        stats.to_csv(file_out, index=False)

    return stats, df_samps

# load data 5cm
uf_05_in ='C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\merged_data_products\\merged_uf_.05m_snow_nearest_canopy_19_149.csv'
uf_05 = pd.read_csv(uf_05_in)

#load data 10cm
uf_10_in ='C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\merged_data_products\\merged_uf_r.10m_canopy_19_149_median-snow.csv'
uf_10 = pd.read_csv(uf_10_in)

# load data 25cm
uf_25_in ='C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\merged_data_products\\merged_uf_r.25m_canopy_19_149_median-snow.csv'
uf_25 = pd.read_csv(uf_25_in)


#####
# sample
d_bounds = [0.05, 30]

stats_swe_045, samps_045 = spatial_stats_on_col(uf_05, 'swe_fcon_19_045', d_bounds, iterations=10000, replicates=120, nbins=60)
stats_swe_045.to_csv(plot_out_dir + "semivar_stats_swe_fcon_045_r.05" + str(d_bounds[1]) + ".csv", index=False)
samps_045.to_csv(plot_out_dir + "semivar_samps_swe_fcon_045_r.05" + str(d_bounds[1]) + ".csv", index=False)

stats_swe_050, samps_050 = spatial_stats_on_col(uf_05, 'swe_fcon_19_050', d_bounds, iterations=10000, replicates=120, nbins=60)
stats_swe_050.to_csv(plot_out_dir + "semivar_stats_swe_fcon_050_r.05_max" + str(d_bounds[1]) + ".csv", index=False)
samps_050.to_csv(plot_out_dir + "semivar_samps_swe_fcon_050_r.05_max" + str(d_bounds[1]) + ".csv", index=False)

stats_swe_052, samps_052 = spatial_stats_on_col(uf_05, 'swe_fcon_19_052', d_bounds, iterations=10000, replicates=120, nbins=60)
stats_swe_052.to_csv(plot_out_dir + "semivar_stats_swe_fcon_052_r.05_max" + str(d_bounds[1]) + ".csv", index=False)
samps_052.to_csv(plot_out_dir + "semivar_samps_swe_fcon_052_r.05_max" + str(d_bounds[1]) + ".csv", index=False)

stats_dswe_045_050, samps_dswe_045_050 = spatial_stats_on_col(uf_05, 'dswe_fnsd_19_045-19_050', d_bounds, iterations=10000, replicates=120, nbins=60)
stats_dswe_045_050.to_csv(plot_out_dir + "semivar_stats_dswe_fnsd_045_050_r.05_max" + str(d_bounds[1]) + ".csv", index=False)
samps_dswe_045_050.to_csv(plot_out_dir + "semivar_samps_dswe_fnsd_045_050_r.05_max" + str(d_bounds[1]) + ".csv", index=False)

stats_dswe_050_052, samps_dswe_050_052 = spatial_stats_on_col(uf_05, 'dswe_fnsd_19_050-19_052', d_bounds, iterations=10000, replicates=120, nbins=60)
stats_dswe_050_052.to_csv(plot_out_dir + "semivar_stats_dswe_fnsd_050_052_r.05_max" + str(d_bounds[1]) + ".csv", index=False)
samps_dswe_050_052.to_csv(plot_out_dir + "semivar_samps_dswe_fnsd_050_052_r.05_max" + str(d_bounds[1]) + ".csv", index=False)

stats_mch, samps_mch = spatial_stats_on_col(uf_10, 'mCH_19_149_resampled', d_bounds, iterations=10000, replicates=120, nbins=60)
stats_mch.to_csv(plot_out_dir + "semivar_stats_mCH_resamp_r.10_max" + str(d_bounds[1]) + ".csv", index=False)
samps_mch.to_csv(plot_out_dir + "semivar_samps_mCH_resamp_r.10_max" + str(d_bounds[1]) + ".csv", index=False)

stats_lrs_cn_1deg, samps_lrs_cn_1deg = spatial_stats_on_col(uf_25, 'lrs_cn_1_deg', d_bounds, iterations=10000, replicates=120, nbins=60)
stats_lrs_cn_1deg.to_csv(plot_out_dir + "semivar_stats_lrs_cn_1_deg_r.25_max" + str(d_bounds[1]) + ".csv", index=False)
samps_lrs_cn_1deg.to_csv(plot_out_dir + "semivar_samps_lrs_cn_1_deg_r.25_max" + str(d_bounds[1]) + ".csv", index=False)

stats_lrs_cn_15deg, samps_lrs_cn_15deg = spatial_stats_on_col(uf_25, 'lrs_cn_1', d_bounds, iterations=10000, replicates=120, nbins=60)
stats_lrs_cn_15deg.to_csv(plot_out_dir + "semivar_stats_lrs_cn_1_r.25_max" + str(d_bounds[1]) + ".csv", index=False)
samps_lrs_cn_15deg.to_csv(plot_out_dir + "semivar_samps_lrs_cn_1_r.25_max" + str(d_bounds[1]) + ".csv", index=False)

stats_lrs_cn_75deg, samps_lrs_cn_75deg = spatial_stats_on_col(uf_25, 'lrs_cn_75_deg', d_bounds, iterations=10000, replicates=120, nbins=60)
stats_lrs_cn_75deg.to_csv(plot_out_dir + "semivar_stats_lrs_cn_75_deg_r.25_max" + str(d_bounds[1]) + ".csv", index=False)
samps_lrs_cn_75deg.to_csv(plot_out_dir + "semivar_samps_lrs_cn_75_deg_r.25_max" + str(d_bounds[1]) + ".csv", index=False)

stats_hemi_lai, samps_hemi_lai = spatial_stats_on_col(uf_25, 'lai_s_cc', d_bounds, iterations=10000, replicates=120, nbins=60)
stats_hemi_lai.to_csv(plot_out_dir + "semivar_stats_lai_hemi_r.25_max" + str(d_bounds[1]) + ".csv", index=False)
samps_hemi_lai.to_csv(plot_out_dir + "semivar_samps_lai_hemi_r.25_max" + str(d_bounds[1]) + ".csv", index=False)

# lai

stats_lrs_lai_1deg, samps_lrs_lai_1deg = spatial_stats_on_col(uf_25, 'lrs_lai_1_deg', d_bounds, iterations=10000, replicates=120, nbins=60)
stats_lrs_lai_1deg.to_csv(plot_out_dir + "semivar_stats_lrs_lai_1_deg_r.25_max" + str(d_bounds[1]) + ".csv", index=False)
samps_lrs_lai_1deg.to_csv(plot_out_dir + "semivar_samps_lrs_lai_1_deg_r.25_max" + str(d_bounds[1]) + ".csv", index=False)

stats_lrs_lai_15deg, samps_lrs_lai_15deg = spatial_stats_on_col(uf_25, 'lrs_lai_15_deg', d_bounds, iterations=10000, replicates=120, nbins=60)
stats_lrs_lai_15deg.to_csv(plot_out_dir + "semivar_stats_lrs_lai_15_deg_r.25_max" + str(d_bounds[1]) + ".csv", index=False)
samps_lrs_lai_15deg.to_csv(plot_out_dir + "semivar_samps_lrs_lai_15_deg_r.25_max" + str(d_bounds[1]) + ".csv", index=False)

stats_lrs_lai_60deg, samps_lrs_lai_60deg = spatial_stats_on_col(uf_25, 'lrs_lai_60_deg', d_bounds, iterations=10000, replicates=120, nbins=60)
stats_lrs_lai_60deg.to_csv(plot_out_dir + "semivar_stats_lrs_lai_60_deg_r.25_max" + str(d_bounds[1]) + ".csv", index=False)
samps_lrs_lai_60deg.to_csv(plot_out_dir + "semivar_samps_lrs_lai_60_deg_r.25_max" + str(d_bounds[1]) + ".csv", index=False)

stats_lrs_lai_2000, samps_lrs_lai_2000 = spatial_stats_on_col(uf_25, 'lrs_lai_2000', d_bounds, iterations=10000, replicates=120, nbins=60)
stats_lrs_lai_2000.to_csv(plot_out_dir + "semivar_stats_lrs_lai_2000_r.25_max" + str(d_bounds[1]) + ".csv", index=False)
samps_lrs_lai_2000.to_csv(plot_out_dir + "semivar_samps_lrs_lai_2000_r.25_max" + str(d_bounds[1]) + ".csv", index=False)

##

stats_re_dswe_045_050, samps_re_dswe_045_050 = spatial_stats_on_col(uf_25, 'dswe_fnsd_19_045-19_050', d_bounds, iterations=10000, replicates=120, nbins=60)
stats_re_dswe_045_050.to_csv(plot_out_dir + "semivar_stats_dswe_fnsd_045_050_r.25_max" + str(d_bounds[1]) + ".csv", index=False)
samps_re_dswe_045_050.to_csv(plot_out_dir + "semivar_samps_dswe_fnsd_045_050_r.25_max" + str(d_bounds[1]) + ".csv", index=False)

stats_re_dswe_050_052, samps_re_dswe_050_052 = spatial_stats_on_col(uf_25, 'dswe_fnsd_19_050-19_052', d_bounds, iterations=10000, replicates=120, nbins=60)
stats_re_dswe_050_052.to_csv(plot_out_dir + "semivar_stats_dswe_fnsd_050_052_r.25_max" + str(d_bounds[1]) + ".csv", index=False)
samps_re_dswe_050_052.to_csv(plot_out_dir + "semivar_samps_dswe_fnsd_050_052_r.25_max" + str(d_bounds[1]) + ".csv", index=False)

# reload sample data
samps_swe_045 = pd.read_csv(plot_out_dir + "semivar_samps_swe_fcon_045_r.05_max" + str(d_bounds[1]) + ".csv")
samps_swe_050 = pd.read_csv(plot_out_dir + "semivar_samps_swe_fcon_050_r.05_max" + str(d_bounds[1]) + ".csv")
samps_swe_052 = pd.read_csv(plot_out_dir + "semivar_samps_swe_fcon_052_r.05_max" + str(d_bounds[1]) + ".csv")
samps_dswe_045_050 = pd.read_csv(plot_out_dir + "semivar_samps_dswe_fnsd_045_050_r.05_max" + str(d_bounds[1]) + ".csv")
samps_dswe_050_052 = pd.read_csv(plot_out_dir + "semivar_samps_dswe_fnsd_050_052_r.05_max" + str(d_bounds[1]) + ".csv")
samps_mch = pd.read_csv(plot_out_dir + "semivar_samps_mCH_resamp_r.10_max" + str(d_bounds[1]) + ".csv")
# samps_lrs_cn_1deg = pd.read_csv(plot_out_dir + "semivar_samps_lrs_cn_1_deg_r.25_max" + str(d_bounds[1]) + ".csv")
# samps_lrs_cn_15deg = pd.read_csv(plot_out_dir + "semivar_samps_lrs_cn_1_r.25_max" + str(d_bounds[1]) + ".csv")
# samps_lrs_cn_75deg = pd.read_csv(plot_out_dir + "semivar_samps_lrs_cn_75_deg_r.25_max" + str(d_bounds[1]) + ".csv")
# samps_hemi_lai = pd.read_csv(plot_out_dir + "semivar_samps_lai_hemi_r.25_max" + str(d_bounds[1]) + ".csv")
samps_lrs_lai_1deg = pd.read_csv(plot_out_dir + "semivar_samps_lrs_lai_1_deg_r.25_max" + str(d_bounds[1]) + ".csv")
samps_lrs_lai_15deg = pd.read_csv(plot_out_dir + "semivar_samps_lrs_lai_15_deg_r.25_max" + str(d_bounds[1]) + ".csv")
samps_lrs_lai_60deg = pd.read_csv(plot_out_dir + "semivar_samps_lrs_lai_60_deg_r.25_max" + str(d_bounds[1]) + ".csv")
samps_lrs_lai_2000 = pd.read_csv(plot_out_dir + "semivar_samps_lrs_lai_2000_r.25_max" + str(d_bounds[1]) + ".csv")
samps_re_dswe_045_050 = pd.read_csv(plot_out_dir + "semivar_samps_dswe_fnsd_045_050_r.25_max" + str(d_bounds[1]) + ".csv")
samps_re_dswe_050_052 = pd.read_csv(plot_out_dir + "semivar_samps_dswe_fnsd_050_052_r.25_max" + str(d_bounds[1]) + ".csv")

# stats_swe_045 = pd.read_csv(plot_out_dir + "semivar_stats_swe_fcon_045_r.05_max" + str(max_dist) + ".csv")
# stats_swe_050 = pd.read_csv(plot_out_dir + "semivar_stats_swe_fcon_050_r.05_max" + str(max_dist) + ".csv")
# stats_swe_052 = pd.read_csv(plot_out_dir + "semivar_stats_swe_fcon_052_r.05_max" + str(max_dist) + ".csv")
# stats_dswe_045_050 = pd.read_csv(plot_out_dir + "semivar_stats_dswe_fnsd_045_050_r.05_max" + str(max_dist) + ".csv")
# stats_dswe_050_052 = pd.read_csv(plot_out_dir + "semivar_stats_dswe_fnsd_050_052_r.05_max" + str(max_dist) + ".csv")
# stats_mch = pd.read_csv(plot_out_dir + "semivar_stats_mCH_resamp_r.10_max" + str(max_dist) + ".csv")
# stats_lrs_cn_1deg = pd.read_csv(plot_out_dir + "semivar_stats_lrs_cn_1_deg_r.25_max" + str(max_dist) + ".csv")
# stats_lrs_cn_15deg = pd.read_csv(plot_out_dir + "semivar_stats_lrs_cn_1_r.25_max" + str(max_dist) + ".csv")
# stats_lrs_cn_75deg = pd.read_csv(plot_out_dir + "semivar_stats_lrs_cn_75_deg_r.25_max" + str(max_dist) + ".csv")
# stats_lai_hemi = pd.read_csv(plot_out_dir + "semivar_stats_lai_hemi_md" + str(max_dist) + ".csv")


# bin and stat
n_bins = 60
d_bounds = [0, 30]

stats_swe_045 = geotk.bin_summarize(samps_swe_045, n_bins, d_bounds=d_bounds, symmetric=True)
stats_swe_050 = geotk.bin_summarize(samps_swe_050, n_bins, d_bounds=d_bounds, symmetric=True)
stats_swe_052 = geotk.bin_summarize(samps_swe_052, n_bins, d_bounds=d_bounds, symmetric=True)
stats_dswe_045_050 = geotk.bin_summarize(samps_dswe_045_050, n_bins, d_bounds=d_bounds, symmetric=True)
stats_dswe_050_052 = geotk.bin_summarize(samps_dswe_050_052, n_bins, d_bounds=d_bounds, symmetric=True)
stats_mch = geotk.bin_summarize(samps_mch, n_bins, d_bounds=d_bounds, symmetric=True)
# stats_lrs_cn_1deg = geotk.bin_summarize(samps_lrs_cn_1deg, n_bins, d_bounds=d_bounds)
# stats_lrs_cn_15deg = geotk.bin_summarize(samps_lrs_cn_15deg, n_bins, d_bounds=d_bounds)
# stats_lrs_cn_75deg = geotk.bin_summarize(samps_lrs_cn_75deg, n_bins, d_bounds=d_bounds)
# stats_hemi_lai = geotk.bin_summarize(samps_hemi_lai, n_bins, d_bounds=d_bounds)
stats_lrs_lai_1deg = geotk.bin_summarize(samps_lrs_lai_1deg, n_bins, d_bounds=d_bounds, symmetric=True)
stats_lrs_lai_15deg = geotk.bin_summarize(samps_lrs_lai_15deg, n_bins, d_bounds=d_bounds, symmetric=True)
stats_lrs_lai_60deg = geotk.bin_summarize(samps_lrs_lai_60deg, n_bins, d_bounds=d_bounds, symmetric=True)
stats_lrs_lai_2000 = geotk.bin_summarize(samps_lrs_lai_2000, n_bins, d_bounds=d_bounds, symmetric=True)
stats_re_dswe_045_050 = geotk.bin_summarize(samps_re_dswe_045_050, n_bins, d_bounds=d_bounds, symmetric=True)
stats_re_dswe_050_052 = geotk.bin_summarize(samps_re_dswe_050_052, n_bins, d_bounds=d_bounds, symmetric=True)

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
plt.title('Standard deviation of SWE with distance\n 14 Feb. 2019, Upper Forest, 5cm resolution, n=1200000')
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
plt.title('Standard deviation of SWE with distance\n 19 Feb. 2019, Upper Forest, 5cm resolution, n=1200000')
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
plt.title('Standard deviation of SWE with distance\n 21 Feb. 2019, Upper Forest, 5cm resolution, n=1200000')
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
plt.title('Standard deviation of $\Delta$SWE with distance\n 14-19 Feb. 2019, Upper Forest, 5cm resolution, n=1200000')
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
plt.title('Standard deviation of $\Delta$SWE with distance\n 19-21 Feb. 2019, Upper Forest, 5cm resolution, n=1200000')
fig.savefig(plot_out_dir + "semivar_dswe_050_052.png")




stats = stats_mch
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# adding axes
axes.plot(stats.mean_dist, stats.stdev)
axes.set_ylim([0, np.nanmax(stats.stdev) * 1.1])
axes.set_xlim([0, np.nanmax(stats.mean_dist) * 1.025])
plt.xlabel('Distance [m]')
plt.ylabel('Standard deviation of mean canopy height [m]')
plt.title('Standard deviation of mean canopy height with distance\n Upper Forest, 10cm resolution, n=1200000')
fig.savefig(plot_out_dir + "semivar_mCH.png")




stats = stats_lrs_cn_1deg
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# adding axes
axes.plot(stats.mean_dist, stats.stdev)
axes.set_ylim([0, np.nanmax(stats.stdev) * 1.1])
axes.set_xlim([0, np.nanmax(stats.mean_dist) * 1.025])
plt.xlabel('Distance [m]')
plt.ylabel('Standard deviation of ray sampled contact number [-]')
plt.title('Standard deviation of 1-degree contact number with distance\n Upper Forest, 25cm resolution, n=12000000')
fig.savefig(plot_out_dir + "semivar_lrs_cn_1deg.png")


stats = stats_lrs_cn_15deg
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# adding axes
axes.plot(stats.mean_dist, stats.stdev)
axes.set_ylim([0, np.nanmax(stats.stdev) * 1.1])
axes.set_xlim([0, np.nanmax(stats.mean_dist) * 1.025])
plt.xlabel('Distance [m]')
plt.ylabel('Standard deviation of ray sampled contact number [-]')
plt.title('Standard deviation of 15-degree contact number with distance\n Upper Forest, 25cm resolution, n=12000000')
fig.savefig(plot_out_dir + "semivar_lrs_cn_15deg.png")


stats = stats_lrs_cn_75deg
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# adding axes
axes.plot(stats.mean_dist, stats.stdev)
axes.set_ylim([0, np.nanmax(stats.stdev) * 1.1])
axes.set_xlim([0, np.nanmax(stats.mean_dist) * 1.025])
plt.xlabel('Distance [m]')
plt.ylabel('Standard deviation of ray sampled contact number [-]')
plt.title('Standard deviation of 75-degree contact number with distance\n Upper Forest, 25cm resolution, n=12000000')
fig.savefig(plot_out_dir + "semivar_lrs_cn_75deg.png")


stats = stats_hemi_lai
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# adding axes
axes.plot(stats.mean_dist, stats.stdev)
axes.set_ylim([0, np.nanmax(stats.stdev) * 1.1])
axes.set_xlim([0, np.nanmax(stats.mean_dist) * 1.025])
plt.xlabel('Distance [m]')
plt.ylabel('Standard deviation of Hemi-photo LAI [-]')
plt.title('Standard deviation of Hemi-photo LAI with distance\n Upper Forest, 25cm resolution, n=12000000')
fig.savefig(plot_out_dir + "semivar_hemi_lai.png")

stats = stats_lrs_lai_1deg
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# adding axes
axes.plot(stats.mean_dist, stats.stdev)
axes.set_ylim([0, np.nanmax(stats.stdev) * 1.1])
axes.set_xlim([0, np.nanmax(stats.mean_dist) * 1.025])
plt.xlabel('Distance [m]')
plt.ylabel('Standard deviation of ray sampled contact number [-]')
plt.title('Standard deviation of 1-degree LAI with distance\n Upper Forest, 25cm resolution, n=12000000')
fig.savefig(plot_out_dir + "semivar_lrs_lai_1deg.png")


stats = stats_lrs_lai_15deg
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# adding axes
axes.plot(stats.mean_dist, stats.stdev)
axes.set_ylim([0, np.nanmax(stats.stdev) * 1.1])
axes.set_xlim([0, np.nanmax(stats.mean_dist) * 1.025])
plt.xlabel('Distance [m]')
plt.ylabel('Standard deviation of ray sampled contact number [-]')
plt.title('Standard deviation of 15-degree LAI with distance\n Upper Forest, 25cm resolution, n=12000000')
fig.savefig(plot_out_dir + "semivar_lrs_lai_15deg.png")


stats = stats_lrs_lai_75deg
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# adding axes
axes.plot(stats.mean_dist, stats.stdev)
axes.set_ylim([0, np.nanmax(stats.stdev) * 1.1])
axes.set_xlim([0, np.nanmax(stats.mean_dist) * 1.025])
plt.xlabel('Distance [m]')
plt.ylabel('Standard deviation of ray sampled contact number [-]')
plt.title('Standard deviation of 75-degree LAI with distance\n Upper Forest, 25cm resolution, n=12000000')
fig.savefig(plot_out_dir + "semivar_lrs_lai_75deg.png")


stats = stats_lrs_lai_2000
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# adding axes
axes.plot(stats.mean_dist, stats.stdev)
axes.set_ylim([0, np.nanmax(stats.stdev) * 1.1])
axes.set_xlim([0, np.nanmax(stats.mean_dist) * 1.025])
plt.xlabel('Distance [m]')
plt.ylabel('Standard deviation of LAI-2000 [-]')
plt.title('Standard deviation of LAI-2000 with distance\n Upper Forest, 25cm resolution, n=12000000')
fig.savefig(plot_out_dir + "semivar_lrs_lai_2000.png")

stats = stats_re_dswe_045_050
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# adding axes
axes.plot(stats.mean_dist, stats.stdev)
axes.set_ylim([0, np.nanmax(stats.stdev) * 1.1])
axes.set_xlim([0, np.nanmax(stats.mean_dist) * 1.025])
plt.xlabel('Distance [m]')
plt.ylabel('Standard deviation of $\Delta$SWE [mm]')
plt.title('Standard deviation of $\Delta$SWE with distance\n 14-19 Feb. 2019, Upper Forest, 25cm resolution, n=1200000')
fig.savefig(plot_out_dir + "semivar_dswe_045_050_r.25.png")

stats = stats_re_dswe_050_052
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# adding axes
axes.plot(stats.mean_dist, stats.stdev)
axes.set_ylim([0, np.nanmax(stats.stdev) * 1.1])
axes.set_xlim([0, np.nanmax(stats.mean_dist) * 1.025])
plt.xlabel('Distance [m]')
plt.ylabel('Standard deviation of $\Delta$SWE [mm]')
plt.title('Standard deviation of $\Delta$SWE with distance\n 19-21 Feb. 2019, Upper Forest, 25cm resolution, n=1200000')
fig.savefig(plot_out_dir + "semivar_dswe_050_052_r.25.png")

# plot semivariograms all together

# calculate set variance
var_swe_045 = np.nanvar(uf_05.swe_fcon_19_045)
var_swe_050 = np.nanvar(uf_05.swe_fcon_19_050)
var_swe_052 = np.nanvar(uf_05.swe_fcon_19_052)
var_dswe_045_050 = np.nanvar(uf_05.loc[:, 'dswe_fnsd_19_045-19_050'])
var_dswe_050_052 = np.nanvar(uf_05.loc[:, 'dswe_fnsd_19_050-19_052'])
var_mch = np.nanvar(uf_10.mCH_19_149_resampled)
# var_lrs_cn_1deg = np.nanvar(uf_25.lrs_cn_1_deg)
# var_lrs_cn_15deg = np.nanvar(uf_25.lrs_cn_1)
# var_lrs_cn_75deg = np.nanvar(uf_25.lrs_cn_75_deg)
# var_hemi_lai = np.nanvar(uf_25.lai_s_cc)
var_lrs_lai_1deg = np.nanvar(uf_25.lrs_lai_1_deg)
var_lrs_lai_15deg = np.nanvar(uf_25.lrs_lai_15_deg)
var_lrs_lai_75deg = np.nanvar(uf_25.lrs_lai_75_deg)
var_lrs_lai_2000 = np.nanvar(uf_25.lrs_lai_2000)
var_re_dswe_045_050 = np.nanvar(uf_25.loc[:, 'dswe_fnsd_19_045-19_050'])
var_re_dswe_050_052 = np.nanvar(uf_25.loc[:, 'dswe_fnsd_19_050-19_052'])

## norm at 30m
norm_swe_045 = stats_swe_045.variance.values[-1]
norm_swe_050 = stats_swe_050.variance.values[-1]
norm_swe_052 = stats_swe_052.variance.values[-1]
norm_dswe_045_050 = stats_dswe_045_050.variance.values[-1]
norm_dswe_050_052 = stats_dswe_050_052.variance.values[-1]
norm_mch = stats_mch.variance.values[-1]
# norm_lrs_cn_1deg = stats_lrs_cn_1deg.variance.values[-1]
# norm_lrs_cn_15deg = stats_lrs_cn_15deg.variance.values[-1]
# norm_lrs_cn_75deg = stats_lrs_cn_75deg.variance.values[-1]
# norm_hemi_lai = stats_hemi_lai.variance.values[-1]
norm_lrs_lai_1deg = stats_lrs_lai_1deg.variance.values[-1]
norm_lrs_lai_15deg = stats_lrs_lai_15deg.variance.values[-1]
norm_lrs_lai_60deg = stats_lrs_lai_60deg.variance.values[-1]
norm_lrs_lai_2000 = stats_lrs_lai_2000.variance.values[-1]
norm_re_dswe_045_050 = stats_re_dswe_045_050.variance.values[-1]
norm_re_dswe_050_052 = stats_re_dswe_050_052.variance.values[-1]

fig = plt.figure()
axes = fig.add_axes([0.13, 0.1, 0.62, 0.8])
# adding axes
plt.xlabel('Distance (d) [m]')
plt.ylabel('Relative variance $\left( \\frac{var(x_{d})}{var(x_{30})}\\right)$ [-]')
plt.title('Relative variance of metrics with distance\n Upper Forest, 5-25cm resolution, n=1200000')

axes.plot(stats_swe_045.mean_dist, .5 * stats_swe_045.variance / norm_swe_045, label="SWE 045", linestyle="dashed")
axes.plot(stats_swe_050.mean_dist, .5 * stats_swe_050.variance / norm_swe_050, label="SWE 050", linestyle="dashed")
axes.plot(stats_swe_052.mean_dist, .5 * stats_swe_052.variance / norm_swe_052, label="SWE 052", linestyle="dashed")

axes.plot(stats_dswe_045_050.mean_dist, .5 * stats_dswe_045_050.variance / norm_dswe_045_050, label="$\Delta$SWE 045-050", linestyle="dotted")
axes.plot(stats_dswe_050_052.mean_dist, .5 * stats_dswe_050_052.variance / norm_dswe_050_052, label="$\Delta$SWE 050-052", linestyle="dotted")

axes.plot(stats_mch.mean_dist, .5 * stats_mch.variance / norm_mch, label="mean canopy height")
# axes.plot(stats_lrs_cn_1deg.mean_dist, .5 * stats_lrs_cn_1deg.variance / var_lrs_cn_1deg, label="$CN_{1}$")
# axes.plot(stats_lrs_cn_15deg.mean_dist, .5 * stats_lrs_cn_15deg.variance / var_lrs_cn_15deg, label="$CN_{15}$")
# axes.plot(stats_lrs_cn_75deg.mean_dist, .5 * stats_lrs_cn_75deg.variance / var_lrs_cn_75deg, label="$CN_{75}$")
# axes.plot(stats_hemi_lai.mean_dist, .5 * stats_hemi_lai.variance / norm_hemi_lai, label="$LAI_{hemi}$")
axes.plot(stats_lrs_lai_1deg.mean_dist, .5 * stats_lrs_lai_1deg.variance / norm_lrs_lai_1deg, label="$LAI_{1}$")
axes.plot(stats_lrs_lai_15deg.mean_dist, .5 * stats_lrs_lai_15deg.variance / norm_lrs_lai_15deg, label="$LAI_{15}$")
axes.plot(stats_lrs_lai_60deg.mean_dist, .5 * stats_lrs_lai_60deg.variance / norm_lrs_lai_60deg, label="$LAI_{75}$")
axes.plot(stats_lrs_lai_2000.mean_dist, .5 * stats_lrs_lai_2000.variance / norm_lrs_lai_2000, label="$LAI_{2000}$")

axes.legend(loc='center left', bbox_to_anchor=(1.01, .5), ncol=1,
            borderaxespad=0, frameon=False)
fig.savefig(plot_out_dir + "semivar_combined_30m.png")


fig = plt.figure()
axes = fig.add_axes([0.13, 0.1, 0.62, 0.8])
# adding axes
plt.xlabel('Distance (d) [m]')
plt.ylabel('Relative variance $\left( \\frac{var(x_{d})}{var(x_{30})}\\right)$ [-]')
plt.title('Relative variance of SWE metrics with distance\n Upper Forest, 5cm resolution (n=1200000)')

axes.plot(stats_swe_045.mean_dist, stats_swe_045.variance / norm_swe_045, label="SWE 14 Feb", linestyle="dashed")
axes.plot(stats_swe_050.mean_dist, stats_swe_050.variance / norm_swe_050, label="SWE 19 Feb", linestyle="dashed")
axes.plot(stats_swe_052.mean_dist, stats_swe_052.variance / norm_swe_052, label="SWE 21 Feb", linestyle="dashed")

axes.plot(stats_dswe_045_050.mean_dist, stats_dswe_045_050.variance / norm_dswe_045_050, label="$\Delta$SWE Storm 1", linestyle="dotted")
axes.plot(stats_dswe_050_052.mean_dist, stats_dswe_050_052.variance / norm_dswe_050_052, label="$\Delta$SWE Storm 2", linestyle="dotted")
axes.legend(loc='center left', bbox_to_anchor=(1.01, .5), ncol=1,
            borderaxespad=0, frameon=False)
fig.savefig(plot_out_dir + "semivar_all_swe_dswe_30m.png")


fig = plt.figure()
axes = fig.add_axes([0.13, 0.1, 0.62, 0.8])
# adding axes
plt.xlabel('Distance (d) [m]')
plt.ylabel('Relative variance $\left( \\frac{var(x_{d})}{var(x_{30})}\\right)$ [-]')
plt.title('Relative variance of canopy metrics with distance\n Upper Forest, 25cm resolution (n=12000000)')

# axes.plot(stats_mch.mean_dist, .5 * stats_mch.variance / var_mch, label="mean canopy height")
# axes.plot(stats_lrs_cn_1deg.mean_dist, .5 * stats_lrs_cn_1deg.variance / var_lrs_cn_1deg, label="$CN_{1}$")
# axes.plot(stats_lrs_cn_15deg.mean_dist, .5 * stats_lrs_cn_15deg.variance / var_lrs_cn_15deg, label="$CN_{15}$")
# axes.plot(stats_lrs_cn_75deg.mean_dist, .5 * stats_lrs_cn_75deg.variance / var_lrs_cn_75deg, label="$CN_{75}$")
axes.plot(stats_lrs_lai_1deg.mean_dist, stats_lrs_lai_1deg.variance / norm_lrs_lai_1deg, label=r"$LAI_{1}^{\blacktriangle}$")
axes.plot(stats_lrs_lai_15deg.mean_dist, stats_lrs_lai_15deg.variance / norm_lrs_lai_15deg, label=r"$LAI_{15}^{\blacktriangle}$")
axes.plot(stats_lrs_lai_60deg.mean_dist, stats_lrs_lai_60deg.variance / norm_lrs_lai_60deg, label=r"$LAI_{60}^{\blacktriangle}$")
axes.plot(stats_lrs_lai_2000.mean_dist, stats_lrs_lai_2000.variance / norm_lrs_lai_2000, label=r"$LAI_{2000}^{\blacktriangle}$")
axes.legend(loc='center left', bbox_to_anchor=(1.01, .5), ncol=1,  borderaxespad=0, frameon=False)
fig.savefig(plot_out_dir + "semivar_lai_30m.png")

# calculate subpixel fractional variance

# 5cm
res = 0.05
n_bins = np.rint((30 * 2 / res)).astype(int)

d_bounds = [0, res * (n_bins)/2]
spv_swe_045 = geotk.bin_summarize(samps_swe_045, n_bins, d_bounds=d_bounds)
spv_swe_045.variance[0] / spv_swe_045.variance[n_bins-1]
spv_swe_050 = geotk.bin_summarize(samps_swe_050, n_bins, d_bounds=d_bounds)
spv_swe_050.variance[0] / spv_swe_050.variance[n_bins-1]
spv_swe_052 = geotk.bin_summarize(samps_swe_052, n_bins, d_bounds=d_bounds)
spv_swe_052.variance[0] / spv_swe_052.variance[n_bins-1]
spv_dswe_045_050 = geotk.bin_summarize(samps_dswe_045_050, n_bins, d_bounds=d_bounds)
spv_dswe_045_050.variance[0] / spv_dswe_045_050.variance[n_bins-1]
(spv_dswe_045_050.variance - spv_dswe_045_050.variance[0]) / (spv_dswe_045_050.variance - spv_dswe_045_050.variance[n_bins - 1])

(stats_dswe_045_050.variance - spv_dswe_045_050.variance[0]) / (spv_dswe_045_050.variance[n_bins - 1] - spv_dswe_045_050.variance[0])


spv_dswe_050_052 = geotk.bin_summarize(samps_dswe_050_052, n_bins, d_bounds=d_bounds)
spv_dswe_050_052.variance[0] / spv_dswe_050_052.variance[n_bins-1]
(spv_dswe_050_052.variance - spv_dswe_050_052.variance[0]) / (spv_dswe_050_052.variance[n_bins - 1] - spv_dswe_050_052.variance[0])
(stats_dswe_050_052.variance - spv_dswe_050_052.variance[0]) / (spv_dswe_050_052.variance[n_bins - 1] - spv_dswe_050_052.variance[0])

# 10cm
res = 0.1
n_bins = np.rint(30 * 2 / res).astype(int)
d_bounds = [0, res * (n_bins)/2]
spv_mch = geotk.bin_summarize(samps_mch, n_bins, d_bounds=d_bounds)
spv_mch.variance[0] / spv_mch.variance[n_bins-2]

# 25cm
res = 0.25
n_bins = np.rint(30 * 2 / res).astype(int)

d_bounds = [0, res * (n_bins)/2]
spv_lrs_lai_1deg = geotk.bin_summarize(samps_lrs_lai_1deg, n_bins, d_bounds=d_bounds)
spv_lrs_lai_1deg.variance[0] / spv_lrs_lai_1deg.variance[n_bins-2]
spv_lrs_lai_15deg = geotk.bin_summarize(samps_lrs_lai_15deg, n_bins, d_bounds=d_bounds)
spv_lrs_lai_15deg.variance[0] / spv_lrs_lai_15deg.variance[n_bins-2]
spv_lrs_lai_75deg = geotk.bin_summarize(samps_lrs_lai_75deg, n_bins, d_bounds=d_bounds)
spv_lrs_lai_75deg.variance[0] / spv_lrs_lai_75deg.variance[n_bins-2]
spv_lrs_lai_2000 = geotk.bin_summarize(samps_lrs_lai_2000, n_bins, d_bounds=d_bounds)
spv_lrs_lai_2000.variance[0] / spv_lrs_lai_2000.variance[n_bins-2]
spv_re_dswe_045_050 = geotk.bin_summarize(samps_re_dswe_045_050, n_bins, d_bounds=d_bounds)
spv_re_dswe_045_050.variance[0] / spv_re_dswe_045_050.variance[n_bins-2]
spv_re_dswe_050_052 = geotk.bin_summarize(samps_re_dswe_050_052, n_bins, d_bounds=d_bounds)
spv_re_dswe_050_052.variance[0] / spv_re_dswe_050_052.variance[n_bins-2]

lala = spv_lrs_lai_1deg.variance / norm_lrs_lai_1deg