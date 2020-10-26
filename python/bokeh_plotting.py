from bokeh.plotting import figure, output_file, show


x = [1, 2, 3, 4, 5]
y = [6, 7, 2, 4, 5]

# output to static HTML file
output_file("C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\graphics\\bokeh\\lines.html")

# create a new plot with a title and axis labels
p = figure(title="simple line example", x_axis_label='x', y_axis_label='y')

# add a line renderer with legend and line thickness
p.line(x, y, legend_label="Temp.", line_width=2)

# show the results
show(p)

# datashader bokeh


### holoviews bokeh
import pandas as pd
import numpy as np
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')


scatter = hv.Scatter(data, 'swe_19_045', 'dce')
scatter

### Datashader

import datashader as ds
import pandas as pd
import datashader.transfer_functions as tf
from datashader.utils import export_image
data_in ='C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\mb_15_merged_.10m_canopy_19_149.csv'
img_out = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\graphics\\test_swe_19_045_vs_dce.png"

data = pd.read_csv(data_in)

cvs = ds.Canvas(plot_width=1000, plot_height=1000)
agg = cvs.points(data, 'swe_19_045', 'dce', agg=ds.count('dce'))
img = tf.shade(agg, cmap=['lightblue', 'darkblue'], how='log')
export_image(img, img_out)

p = figure(title="simple line example", x_axis_label='x', y_axis_label='y')

# add a line renderer with legend and line thickness
p.image(img)

# show the results
show(p)