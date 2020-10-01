import rastools

ras_1_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\qc_test\\19_149_dem_r.25m_count_crop1.tif"
ras_2_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\qc_test\\19_149_dem_r.25m_count_crop2.tif"
ras_3_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\qc_test\\19_149_dem_r.25m_count_crop3.tif"
ras_4_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\qc_test\\19_149_dem_r.25m_count_crop4.tif"

data_1 = rastools.pd_sample_raster(None, None, ras_1_in, 'count_1', include_nans=False)
data_re = rastools.pd_sample_raster(data_1, ras_1_in, ras_2_in, 'count_2', include_nans=False)
data_2 = rastools.pd_sample_raster(None, ras_2_in, 'count_2', include_nans=False)

# i trust that the parent index values are accurate. The issue is not with the raster_to_pd function

data_3 = rastools.pd_sample_raster(None, ras_3_in, 'count_3', include_nans=False)
data_re = rastools.pd_sample_raster(data_3, ras_4_in, 'count_4', include_nans=False)
data_4 = rastools.pd_sample_raster(None, ras_4_in, 'count_4', include_nans=False)

data_re = rastools.pd_sample_raster(data_3, ras_1_in, 'count_1', include_nans=False)

parent = data_1
ras = ras_2_in
colnames='count_2'
include_nans = False

# it appears that the x
# check out the image of both...
import numpy as np
data = data_4
img = np.full((np.max(data.y_index) + 1, np.max(data.x_index) + 1), np.nan)
img[data.y_index, data.x_index] = data.count_4

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.imshow(img, interpolation='nearest')

ras = rastools.raster_load(ras_1_in)

# [y_index, x_index] = ~T * [x_coord, y_coord]
train = ~ras.T1 * (data_1.x_coord, data_1.y_coord)
np.max(np.array(data_1.x_index) - np.array(train[1]))
np.max(np.array(data_1.y_index) - np.array(train[0]))

# [x_coord, y_coord] = T * [y_index, x_index]
peace = ras.T1 * (data_1.y_index, data_1.x_index)
np.all(np.array(data_1.x_coord) == np.array(peace[0]))
np.all(np.array(data_1.y_coord) == np.array(peace[1]))



