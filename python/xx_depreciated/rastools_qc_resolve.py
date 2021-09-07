from libraries import raslib
import numpy as np

parent = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\DEM\\19_149_dem_r.25m_count.tif"
ras_1_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\qc_test\\19_149_dem_r.25m_count_crop1.tif"
ras_2_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\qc_test\\19_149_dem_r.25m_count_crop2.tif"
ras_3_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\qc_test\\19_149_dem_r.25m_count_crop3.tif"
ras_4_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\qc_test\\19_149_dem_r.25m_count_crop4.tif"
ras_5_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\qc_test\\1m_dem_point_ids_crop1.tif"
ras_6_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\qc_test\\1m_dem_point_ids_crop2.tif"

files = [ras_1_in, ras_2_in, ras_3_in, ras_4_in, ras_5_in, ras_6_in]
colnames = ['count_1', 'count_2', 'count_3', 'count_4', 'id1', 'id2']

# results agree when running the same large matrix twice
data_1 = raslib.pd_sample_raster(None, None, ras_1_in, 'count_1', include_nans=False)
data_re = raslib.pd_sample_raster(data_1, ras_1_in, ras_1_in, 'count_2', include_nans=False)


data_re = raslib.pd_sample_raster(data_1, ras_1_in, ras_2_in, 'count_2', include_nans=False)
data_2 = raslib.pd_sample_raster(None, ras_2_in, 'count_2', include_nans=False)

# i trust that the parent index values are accurate. The issue is not with the raster_to_pd function

# results agree when running the same raster twice
data_3 = raslib.pd_sample_raster(None, None, ras_3_in, 'count_3', include_nans=False)
data_re = raslib.pd_sample_raster(data_3, ras_3_in, ras_3_in, 'count_3_again', include_nans=False)
np.all(data_re.count_3 == data_re.count_3_again)

# non-nans agree when running all values here
data_re = raslib.pd_sample_raster(data_3, ras_3_in, ras_4_in, 'count_4', include_nans=False)
np.all(data_re.count_3 == data_re.count_4)

data_4 = raslib.pd_sample_raster(None, ras_4_in, 'count_4', include_nans=False)

data_re = raslib.pd_sample_raster(data_3, ras_1_in, 'count_1', include_nans=False)

parent = data_3
ras_parent = ras_3_in
ras_child = ras_3_in
colnames='count_3'
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

ras = raslib.raster_load(ras_1_in)

# [y_index, x_index] = ~T * [x_coord, y_coord]
data_1 = raslib.raster_to_pd(ras_1_in, 'count_1')

train = ~ras.T1 * (data_1.x_coord, data_1.y_coord)
np.max(np.array(data_1.x_index) - np.array(train[1]))
np.max(np.array(data_1.y_index) - np.array(train[0]))

# [x_coord, y_coord] = T * [y_index, x_index]
peace = ras.T1 * (data_1.x_index, data_1.y_index)
np.all(np.array(data_1.x_coord) == np.array(peace[0]))
np.all(np.array(data_1.y_coord) == np.array(peace[1]))


# plot of count_1 and count_2 in parent index

data_5 = raslib.pd_sample_raster(None, None, ras_5_in, 'id1', include_nans=False)
data_re = raslib.pd_sample_raster(data_5, ras_5_in, ras_6_in, 'id2', include_nans=False)

parent = data_5
ras_parent = ras_5_in
ras_child = ras_6_in
colnames='id2'
include_nans = False

x_max = np.max(data_re.x_index)
y_max = np.max(data_re.y_index)

ras_1 = np.full((y_max + 1, x_max + 1), np.nan)
ras_1[data_re.y_index, data_re.x_index] = data_re.id1
plt.imshow(ras_1, interpolation='nearest')

ras_2 = np.full((y_max + 1, x_max + 1), np.nan)
ras_2[data_re.y_index, data_re.x_index] = data_re.id2
plt.imshow(ras_2, interpolation='nearest')


data = raslib.raster_to_pd(ras_1_in, colnames='count_1')
ras_2 = raslib.gdal_raster_reproject(ras_2_in, ras_1_in)
data.loc[:, 'count_2'] = ras_2[data.y_index, data.x_index]

multi_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\RSM\\19_149_expected_returns_res_.25m_0-0_t_1.tif'
ras_m = raslib.gdal_raster_reproject(multi_in, ras_1_in)
ras_m = raslib.gdal_raster_reproject(ras_2_in, ras_1_in)