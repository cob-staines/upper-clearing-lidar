from libraries import raslib
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
from scipy.stats import pearsonr

# plot point density vs. snow depth

ddict = {'uf': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\hemi_grid_points\\mb_65_r.25m\\uf_plot_r.10m.tif',
         "19_045_pdens": "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_045\\19_045_las_proc\\OUTPUT_FILES\\RAS\\19_045_ground_point_density_r.10m.bil",
         "19_149_pdens": "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\RAS\\19_149_ground_point_density_r.10m.bil",
         '19_149_lpml15': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\LPM\\19_149_LPM-last_a15_r0.10m.tif',
         "19_045_hs": "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_045\\19_045_las_proc\\OUTPUT_FILES\\HS\\interp_2x\\clean\\19_045_hs_r.05m_interp2x_clean.tif"}

df = raslib.pd_sample_raster_gdal(ddict, include_nans=False, mode="median")
# sns.scatterplot(data=df, x="19_045_pdens", y="19_045_hs")
x = df.loc[:, "19_045_pdens"] * 0.1 ** 2
y = df.loc[:, "19_045_hs"]

# plt.scatter(x_var, y_var, alpha=0.05)

x_step = 1
plotrange = [np.rint(np.array([0, np.nanquantile(x, .95)]) / x_step) * x_step,
             [0, np.nanquantile(y, .95)]]
n_bins = int((plotrange[0][1] - plotrange[0][0]) / x_step) - 1
rbins = (n_bins, np.rint(n_bins * 5.7/8).astype(int)*8)

plt.hist2d(x, y, range=plotrange, bins=rbins, cmap="Blues")


# with lpml15
x = df.loc[:, "19_149_lpml15"]
y = df.loc[:, "19_045_hs"]

# plt.scatter(x_var, y_var, alpha=0.05)

rbins = (np.array([8, 5.7]) * 20).astype(int)
plotrange = [[np.nanquantile(x, .0005), np.nanquantile(x, .9995)],
             [np.nanquantile(y, .0005), np.nanquantile(y, .9995)]]

plt.hist2d(x, y, range=plotrange, bins=rbins, cmap="Blues")

df.loc[valid, ["19_045_pdens", "19_045_hs"]]
valid = ~np.isnan(x) & ~np.isnan(y)
res = pearsonr(x[valid], y[valid])