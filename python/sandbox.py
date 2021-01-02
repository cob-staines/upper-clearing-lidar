import numpy as np
import pandas as pd

import holoviews as hv
from holoviews import dim, opts
hv.extension('bokeh')

import datashader as ds
# from datashader import transfer_functions as tf

# load data
data_in ='C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\mb_15_merged_.25m_canopy_19_149.csv'
data = pd.read_csv(data_in)

# filter data
# data = data.loc[~np.isnan(data.uf.values), :]

# calculate columns
data.loc[:, 'cn_mean'] = data.loc[:, 'er_p0_mean'] * 0.19546
data.loc[:, 'cn_sd'] = data.loc[:, 'er_p0_sd'] * 0.19546
data.loc[:, ['cn_mean_ln']] = np.log(data.cn_mean)
data.loc[:, ['cn_mean_exp']] = np.exp(-data.cn_mean)
data.loc[:, ['cn_sd_ln']] = data.cn_sd * data.cn_mean_ln / data.cn_mean
data.loc[:, ['lai_ln']] = np.log(data.lai_s_cc)
data.loc[:, ['lai_exp']] = np.exp(-data.lai_s_cc)
data.loc[:, ['cn_weighted_scaled']] = data.cn_weighted * 0.19546
data.loc[:, ['cn_weighted_ln']] = np.log(data.cn_weighted)

data.loc[:, ['1_dswe']] = data.loc[:, ['dswe_19_045-19_052']].values
data.loc[:, ['2_dce']] = data.loc[:, ['dce']].values
data.loc[:, ['3_chm']] = data.loc[:, ['chm']].values
data.loc[:, ['4_hemi_GF']] = data.loc[:, ['lai_exp']].values
data.loc[:, ['5_rs_GF']] = data.loc[:, ['cn_mean_exp']].values

# plot variables
x_var = ['1_dswe', '2_dce', '3_chm', '4_hemi_GF', '5_rs_GF']
# y_var = x_var
y_var = ['1__dswe', '2__dce', '3__chm', '4__hemi_GF', '5__rs_GF']
# x_var = ['dswe_19_045-19_052']
# y_var = ['dswe_19_045-19_052_1']

data.loc[:, y_var] = data.loc[:, x_var].values
# x_var = ['dce']
# x_var = ['dswe_19_045-19_052']
# y_var = ['cn_weighted_ln', 'cn_weighted']
# y_var = ['cn_mean', 'lai_s_cc', 'cn_mean_ln', 'lai_ln']
# y_var = ['cn_mean_ln', 'cn_mean_exp', 'lai_ln', 'lai_exp', 'cn_weighted_scaled', 'contactnum_1', 'dce']

# plot dimentions
plot_width = int(200)
plot_height = plot_width
# plot_height = int(plot_width//1.2)
zoom_multiplier = 1

def covar_plot(x_var, y_var, x_range, y_range):
    cvs = ds.Canvas(plot_width=plot_width, plot_height=plot_height, x_range=x_range, y_range=y_range)
    agg = cvs.points(data, x_var, y_var, ds.count(y_var))
    return hv.Image(agg).opts(colorbar=False, cmap='Blues', logz=True, width=int(plot_width*zoom_multiplier), height=int(plot_height*zoom_multiplier))

plot_dict=hv.OrderedDict([])
for yy in range(0, len(y_var)):
    # y_range = (np.nanmin(data.loc[:, yy]),np.nanmax(data.loc[:, yy]))
    y_range = (np.nanquantile(data.loc[:, y_var[yy]], .001), np.nanquantile(data.loc[:, y_var[yy]], .999))
    for xx in range(yy, len(x_var)):
        # x_range = (np.nanmin(data.loc[:, x_var]),np.nanmax(data.loc[:, x_var]))
        x_range = (np.nanquantile(data.loc[:, x_var[xx]], .001), np.nanquantile(data.loc[:, x_var[xx]], .999))
        plot_dict[(y_var[yy], x_var[xx])] = covar_plot(y_var[yy], x_var[xx], y_range, x_range)


kdims = [hv.Dimension(('y_var', 'yy')),
         hv.Dimension(('x_var', 'xx'))]
holomap = hv.HoloMap(plot_dict, kdims=kdims)
# holomap.opts(opts.Curve(width=600))

grid = hv.GridSpace(holomap, sort=False)
grid