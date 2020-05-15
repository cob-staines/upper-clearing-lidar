import rastools
import vaex
import matplotlib.pylab as plt
import numpy as np

# products to import

# snow depth .10m
hs_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\hs\\19_045\\hs_19_045_res_.10m.tif"
dft_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\DFT\\19_149_all_200311_628000_5646525_spike_free_chm_.10m_kho_distance_.25m.tif"

hs_hdf5 = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\hs\\19_045\\hs_19_045_res_.10m.hdf5"

# hs = rastools.raster_load(hs_in)
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
rastools.hdf5_sample_raster(hs_hdf5, hs_hdf5, [dft_in, site_raster_path], sample_col_name=["dft", "uf"])

##### Plotting #####

df = vaex.open(hs_hdf5, 'r')
no_data = -9999

# selections
hs_na = df.hs.values == no_data
dft_na = df.dft.values == no_data
uf = df.uf == 1

count_x_all = df.count(binby=df.dft, limits=[0, 40], shape=100, selection=(uf & ~dft_na))/(df.count(selection=(uf & ~dft_na)))
count_x_sampled = df.count(binby=df.dft, limits=[0, 40], shape=100, selection=(~dft_na & ~hs_na))/(df.count(selection=(~dft_na & ~hs_na)))
plt.plot(np.linspace(0, 40, 100), count_x_all)
plt.plot(np.linspace(0, 40, 100), count_x_sampled)
plt.show()


df.select(df.x > df.y)

df.close()

###
