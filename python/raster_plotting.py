import rastools
import vaex
import matplotlib.pylab as plt
import numpy as np

# products to import

# snow depth .10m
hs_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\hs\\19_045\\hs_19_045_res_.04m.tif"
dft_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\DFT\\19_149_all_200311_628000_5646525_spike_free_chm_.10m_kho_distance_.25m.tif"

hs_hdf5 = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\hs\\19_045\\hs_19_045_res_.04m.hdf5"
hs_dft_hdf5 = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\hs\\19_045\\hs_19_045_res_.04m._dft_hdf5"

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
rastools.hdf5_sample_raster(hs_hdf5, hs_dft_hdf5, [dft_in, site_raster_path], sample_col_name=["dft", "uf"])
hdf5_in = hs_hdf5
hdf5_out = hs_dft_hdf5
ras_in = [dft_in, site_raster_path]
sample_col_name = ["dft", "uf"]

def hdf5_sample_raster(hdf5_in, hdf5_out, ras_in, sample_col_name="sample"):
    # can be single ras_in/sample_col_name or list of both
    import numpy as np
    import vaex

    if (type(ras_in) == str) & (type(sample_col_name) == str):
        # convert to list of length 1
        ras_in = [ras_in]
        sample_col_name = [sample_col_name]
    elif (type(ras_in) == list) & (type(sample_col_name) == list):
        if len(ras_in) != len(sample_col_name):
            raise Exception('Lists of "ras_in" and "sample_col_name" are not the same length.')
    else:
        raise Exception('"ras_in" and "sample_col_name" are not consistent in length or format.')

    # load hdf5_in
    #df = vaex.open(hdf5_in, 'r+')
    df = vaex.open(hdf5_in)

    for ii in range(0, len(ras_in)):
        # load raster
        ras = rastools.raster_load(ras_in[ii])

        # convert sample points to index refference
        row_col_pts = np.floor(~ras.T0 * (df.UTM11N_x.values, df.UTM11N_y.values)).astype(int)
        #row_col_pts = (row_col_pts[0], row_col_pts[1])

        # flag samples out of raster bounds
        outbound_x = (row_col_pts[0] < 0) | (row_col_pts[0] > (ras.rows - 1))
        outbound_y = (row_col_pts[1] < 0) | (row_col_pts[1] > (ras.cols - 1))
        outbound = outbound_x | outbound_y

        # list of points in bounds
        sample_pts = (row_col_pts[0][~outbound], row_col_pts[1][~outbound])

        # read raster values of sample_points
        samples = np.full(outbound.shape, ras.no_data)
        samples[~outbound] = ras.data[sample_pts]

        # add column to df
        df.add_column(sample_col_name[ii], samples, dtype=None)

        ras = None

    # save to hdf5_out
    df.export_hdf5(hdf5_out)
    df.close()

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
plt.show()

hs_samp.plot(hs_samp.dft, hs_samp.hs)

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