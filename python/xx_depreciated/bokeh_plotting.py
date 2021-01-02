# from bokeh.plotting import figure, output_file, show
#
#
# x = [1, 2, 3, 4, 5]
# y = [6, 7, 2, 4, 5]
#
# # output to static HTML file
# output_file("C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\graphics\\bokeh\\lines.html")
#
# # create a new plot with a title and axis labels
# p = figure(title="simple line example", x_axis_label='x', y_axis_label='y')
#
# # add a line renderer with legend and line thickness
# p.line(x, y, legend_label="Temp.", line_width=2)
#
# # show the results
# show(p)
#
# # datashader bokeh
#
#
# ### holoviews bokeh
# import pandas as pd
# import numpy as np
# import holoviews as hv
# from holoviews import opts
# hv.extension('bokeh')
#
#
# scatter = hv.Scatter(data, 'swe_19_045', 'dce')
# scatter
#
# ### Datashader
#
# import datashader as ds
# import pandas as pd
# import datashader.transfer_functions as tf
# from datashader.utils import export_image
# data_in ='C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\mb_15_merged_.10m_canopy_19_149.csv'
# img_out = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\graphics\\test_swe_19_045_vs_dce.png"
#
# data = pd.read_csv(data_in)
#
# cvs = ds.Canvas(plot_width=1000, plot_height=1000)
# agg = cvs.points(data, 'swe_19_045', 'dce', agg=ds.count('dce'))
# img = tf.shade(agg, cmap=['lightblue', 'darkblue'], how='log')
# export_image(img, img_out)
#
# p = figure(title="simple line example", x_axis_label='x', y_axis_label='y')
#
# # add a line renderer with legend and line thickness
# p.image(img)
#
# # show the results
# show(p)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import datashader as ds
import datashader.transfer_functions as tf
from datashader.colors import Hot


img_width = 800
img_height = int(img_width // 1.2)

data_in ='C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\mb_15_merged_.10m_canopy_19_149.csv'
data = pd.read_csv(data_in)

z_var = 'swe_19_045'
# def bin_data():
#     global time_period, grouped, group_count, counter, times, groups
#     grouped = df.groupby([times.hour, times.minute // time_period])
#     groups = sorted(grouped.groups.keys(), key=lambda r: (r[0], r[1]))
#     group_count = len(groups)
#     counter = 0


class LiveImageDisplay(object):
    def __init__(self, h=500, w=500, niter=50, radius=2., power=2):
        self.height = h
        self.width = w

    def __call__(self, xstart, xend, ystart, yend):
        cvs = ds.Canvas(plot_width=self.width,
                           plot_height=self.height,
                           x_range=(xstart, xend),
                           y_range=(ystart, yend))

        agg = cvs.points(data, 'x_coord', 'y_coord', ds.count(z_var))

        # img = tf.shade(agg, cmap='Blues', how='log')
        img = tf.shade(agg, cmap='gray')

        print(img.data.shape, img.data.dtype)
        return img.data

    # def ax_update(self, ax):
    #     ax.set_autoscale_on(False)  # Otherwise, infinite loop
    #
    #     # Get the number of points from the number of pixels in the window
    #     dims = ax.patch.get_window_extent().bounds
    #     self.width = int(dims[2] + 0.5)
    #     self.height = int(dims[2] + 0.5)
    #
    #     # Get the range for the new area
    #     xstart, ystart, xdelta, ydelta = ax.viewLim.bounds
    #     xend = xstart + xdelta
    #     yend = ystart + ydelta
    #
    #     # Update the image object with our new data and extent
    #     im = ax.images[-1]
    #     im.set_data(self.__call__(xstart, xend, ystart, yend))
    #     im.set_extent((xstart, xend, ystart, yend))
    #     ax.figure.canvas.draw_idle()





xmin = np.nanmin(data.x_coord)
ymin = np.nanmin(data.y_coord)
xmax = np.nanmax(data.x_coord)
ymax = np.nanmax(data.y_coord)

img = LiveImageDisplay(w=img_width, h=img_height)
Z = img(xmin, xmax, ymin, ymax)

fig1, ax2 = plt.subplots(1, 1)
img = ax2.imshow(Z, origin='lower', extent=(xmin, xmax, ymin, ymax))
fig1.colorbar(img, ax=ax2)


# Connect for changing the view limits
# ax2.callbacks.connect('xlim_changed', img.ax_update)
# ax2.callbacks.connect('ylim_changed', img.ax_update)

plt.show()