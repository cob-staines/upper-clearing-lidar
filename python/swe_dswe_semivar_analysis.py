import geotk
import rastools

# sample density functions
# def log10_inv(dd):
#     return -np.log10(1 - dd)
#
# def log_inv(dd):
#     return -np.log(1 - dd)

def linear_ab(dd):
    a = 0
    b = 5
    return (b - a) * dd + a

resolution = ['.04', '.10', '.25', '.50', '1.00']
snow_on = ["19_045", "19_050", "19_052", "19_107", "19_123"]

def spatial_stats_on_swe_uf(ras_in):
    ddict = {'swe': ras_in,
             'uf': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\hemi_grid_points\\uf_plot_over_dem.tiff'}

    swe = rastools.pd_sample_raster_gdal(ddict)

    # filter to uf site
    swe = swe[swe.uf == 1]

    # define points and values
    pts = swe.loc[:, ['x_coord', 'y_coord']].values
    vals = swe.swe.values

    # define sample density function
    def linear_ab(x):
        a = 0
        b = 10
        return (b - a) * x + a

    df, unif_bounds = geotk.pnt_sample_semivar(pts, vals, linear_ab, 1000, 100)
    stats = geotk.bin_summarize(df, linear_ab, unif_bounds, 50)

    # export to csv
    stats_out = swe_in.replace('.tif', '_spatial_stats.csv')
    stats.to_csv(stats_out, index=False)

for dd in snow_on:
    for rr in resolution:
        swe_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\SWE\\' + dd + '\\swe_' + dd + '_r' + rr + 'm_q0.25.tif'
        spatial_stats_on_swe_uf(swe_in)



for ii in range(0, len(snow_on) - 1):
    ddi = snow_on[ii]
    ddj = snow_on[ii + 1]

    for rr in resolution:
        dswe_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\dSWE\\" + ddi + "-" + ddj + "\\dswe_" + ddi + "-" + ddj + "_r" + rr + "m_q0.25.tif"
        spatial_stats_on_swe_uf(dswe_in)

#
# import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# plt.scatter(stats.mean_dist, stats.stdev)
# plt.ylim(0, np.max(stats.stdev))
# plt.xlim(0, np.max(stats.mean_dist))
# # plt.show()
#
# plt.scatter(stats.bin_mid, stats.mean_bias)
# plt.scatter(stats.bin_mid, stats.n)
#
# plt.scatter(df.dist, df.dvals)
