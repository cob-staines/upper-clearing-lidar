import rastools
import vaex
import matplotlib.pylab as plt
import numpy as np


# products to import

# snow depth .10m
hs_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\hs\\19_045\\hs_19_045_res_.10m_test.tif"
dft_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\DFT\\19_149_all_200311_628000_5646525_spike_free_chm_.10m_kho_distance_.10m.tif"
dem_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\DEM\\19_149_all_200311_628000_5646525dem_.10m.bil"

hs_hdf5 = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\hs\\19_045\\hs_19_045_res_.04m.hdf5"
hs_dft_hdf5 = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\hs\\19_045\\hs_19_045_res_.04m._dft.hdf5"

hs = rastools.raster_load(hs_in)
# dft = rastools.raster_load(dft_in)

# send hs to hdf5
rastools.raster_to_hdf5(hs_in, hs_hdf5, "hs")

# sample site
# create raster of false values
site_shp_path = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\site_library\\upper_forest_poly_UTM11N.shp"
site_raster_path ="C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\site_library\\upper_forest_poly_UTM11N.tif"

template = rastools.raster_load(hs_in)
template.data = np.full([template.rows, template.cols], 0)
template.no_data = 0
rastools.raster_save(template, site_raster_path, data_format="int16")
# burn in upper forest site as true values
rastools.raster_burn(site_raster_path, site_shp_path, 1)

# sample rasters
rastools.hdf5_sample_raster(hs_hdf5, hs_dft_hdf5, [dft_in, site_raster_path, dem_in], sample_col_name=["dft", "uf", "dem"])

##### Plotting #####
df = vaex.open(hs_dft_hdf5, 'r')
df.get_column_names()
no_data = -9999

df_uf = df[df.uf == 1]

dft = df_uf[df_uf.dft != no_data]
count_dft_all = dft.count(binby=dft.dft, limits=[0, 5], shape=100)/dft.length()
hs_samp = dft[dft.hs != no_data]
count_dft_sampled = hs_samp.count(binby=hs_samp.dft, limits=[0, 5], shape=100)/hs_samp.length()


plt.plot(np.linspace(0, 5, 100), count_dft_all)
plt.plot(np.linspace(0, 5, 100), count_dft_sampled)
# difference
plt.plot(np.linspace(0, 5, 100), count_dft_all - count_dft_sampled)

plt.show()

hs_samp.plot(hs_samp.dft, hs_samp.hs, shape=300, vmax=0.6)

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