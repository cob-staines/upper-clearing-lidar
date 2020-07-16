import rastools
import vaex
import matplotlib.pylab as plt
import numpy as np


# products to import

# snow depth .10m
hs_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\hs\\19_045\\hs_19_045_res_.04m.tif"
dft_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\DNT\\19_149_snow_off_627975_5646450_spike_free_chm_.10m_kho_distance_.10m.tif"
hs_10_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\hs\\19_045\\hs_19_045_res_.10m.tif"

hs_hdf5 = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\hs\\19_045\\hs_19_045_res_.04m.hdf5"
hs_dft_hdf5 = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\hs\\19_045\\hs_19_045_res_.04m._dft.hdf5"

hs = rastools.raster_load(hs_in)
# dft = rastools.raster_load(dft_in)

# send hs to hdf5
rastools.raster_to_hdf5(hs_in, hs_hdf5, "hs_04m")

# sample site
# create raster of false values
site_shp_path = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\site_library\\sub_plot_library\\forest_upper.shp"
site_raster_path ="C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\site_library\\upper_forest_poly_UTM11N.tif"

template = rastools.raster_load(hs_in)
template.data = np.full([template.rows, template.cols], 0)
template.no_data = 0
rastools.raster_save(template, site_raster_path, data_format="int16")
# burn in upper forest site as true values
rastools.raster_burn(site_raster_path, site_shp_path, 1)

# sample rasters
rastools.hdf5_sample_raster(hs_hdf5, hs_dft_hdf5, [dft_in, site_raster_path, hs_10_in], sample_col_name=["dft", "uf", "hs_01m"])

##### Plotting #####
df = vaex.open(hs_dft_hdf5, 'r')
df.get_column_names()
no_data = -9999

df_uf = df[df.uf == 1]

dft = df_uf[df_uf.dft != no_data]
count_dft_all = dft.count(binby=dft.dft, limits=[0, 5], shape=100)/dft.length()
hs_samp = dft[dft.hs_04m != no_data]
count_dft_sampled = hs_samp.count(binby=hs_samp.dft, limits=[0, 5], shape=100)/hs_samp.length()

fig, ax = plt.subplots()
ax.plot(np.linspace(0, 5, 100), count_dft_all)
ax.plot(np.linspace(0, 5, 100), count_dft_sampled)
ax.set(xlabel='Relative frequency', ylabel='Distance to Nearest Tree -- DNT (m)',
       title='Normalized distributions of DNT for all points (blue) and snow-depth sampled points (orange)')
ax.grid()
plt.show()

# difference
plt.plot(np.linspace(0, 5, 100), count_dft_all - count_dft_sampled)
plt.show()

hs_samp.plot(hs_samp.dft, hs_samp.hs__04m, shape=300, vmax=0.6)
hs_samp.plot(hs_samp.hs_04m, hs_samp.hs_01m)

count_hs_sampled = hs_samp.count(binby=hs_samp.hs, limits=[0, 0.6], shape=1000)/hs_samp.length()
plt.plot(np.linspace(0, 0.6, 1000), count_hs_sampled)
count_dem_sampled = hs_samp.count(binby=hs_samp.dem, limits=[1828, 1838], shape=10000)/hs_samp.length()
plt.plot(np.linspace(1828, 1838, 10000), count_dem_sampled)

df.close()

###
hs_ras = df.hs.values.reshape([hs.rows, hs.cols])
dft_ras = df.dft.values.reshape([hs.rows, hs.cols])
dft_ras[dft_ras == -9999] = 0
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
fig1 = plt.imshow(hs_ras, interpolation='nearest')
fig2 = plt.imshow(dft_ras, interpolation='nearest')

###
# LPM plotting
import rastools
import holoviews as hv
import datashader as ds
import datashader.transfer_functions as tf
import holoviews.operation.datashader as hd
import numpy as np
import pandas as pd

hv.extension("bokeh", "matplotlib")
hv.output(backend="matplotlib")


hs_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\hs\\19_045\\hs_19_045_res_.25m.tif"
lpm_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\LPM\\19_149_snow_off_LPM-first_30degsa_0.25m.tif"

# load images
hs = rastools.raster_load(hs_in)
lpmf = rastools.raster_load(lpm_in)

# check projections
hs.T0
lpmf.T0

# define inputs
parent = hs
child = lpmf
# create template of sample points
samples = parent.data.copy()

# create affine transform
x, y = np.ogrid[0:parent.rows, 0:parent.cols]
parentindex = np.where(np.full_like(parent.data, True))
geocoords = parent.T1 * (parentindex[0], parentindex[1])
childindex = np.rint(~child.T1 * (geocoords[0], geocoords[1])).astype(int)
in_bounds = (childindex[0] >= 0) & (childindex[0] < child.rows) & (childindex[1] >= 0) & (childindex[1] < child.cols)
child_in_bounds = (childindex[0][in_bounds], childindex[1][in_bounds])
sample_values = child.data[child_in_bounds]

reshape_values = np.full_like(parentindex[0], parent.no_data)
reshape_values[in_bounds] = sample_values
child_reshaped = np.reshape(reshape_values, parent.data.shape)

plot_data = pd.DataFrame({"hs": parent.data.reshape(parent.rows*parent.cols),
                          "lpmf": child_reshaped.reshape(parent.rows*parent.cols)})

plot_data[plot_data == parent.no_data] = np.nan

cvs = ds.Canvas(plot_width=400, plot_height=400)
agg = cvs.points(plot_data, "hs", "lpmf", ds.count())
hd.datashade(plot_data)
hd.shade(hv.Image(agg))
hv.RGB(np.array(tf.shade(agg).to_pil()))

tf.Image(tf.shade(agg))

import rastools
import numpy as np
import pandas as pd
import hvplot.pandas
import holoviews as hv
hv.extension('bokeh')
from bokeh.plotting import figure, output_file, show

output_file('test_bokeh.html')
p = figure(plot_width=400, plot_height=400)
p.vbar(x=[1, 2, 3], width=0.5, bottom=0, top=[1.2, 2.5, 3.6], color='red')
plot = plot_data.hvplot(kind='scatter', x='hs', y='lpmf', datashade=True)
show(hv.render(plot))
show(p)

# test bokeh
import numpy as np
import pandas as pd
import hvplot.pandas
import holoviews as hv
hv.extension("bokeh")
import bokeh
from bokeh.plotting import show

data = np.random.normal(size=[50, 2])
df = pd.DataFrame(data = data, columns=['col1', 'col2'])

plot = df.hvplot(kind="scatter", x="col1", y="col2")
show(hv.render(plot))
