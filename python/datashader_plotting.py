# basic datashader
import datashader as ds
import pandas as pd
import datashader.transfer_functions as tf
from datashader.utils import export_image

data_in ='C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\mb_15_merged_.25m_canopy_19_149.csv'
img_out = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\graphics\\test_swe_19_045_vs_lai.png"

data = pd.read_csv(data_in)

cvs = ds.Canvas(plot_width=1000, plot_height=1000)
agg = cvs.points(data, 'swe_19_045', 'lai_s_cc', agg=ds.count('dnt'))
img = tf.shade(agg, cmap=['lightblue', 'darkblue'], how='log')
export_image(img, img_out)




# basic datashader
# import rastools
#
# import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
#
# # plt.scatter(data.swe_19_045, data.dnt)
