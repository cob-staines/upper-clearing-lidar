# import rasterio
# import ogr


def raster_load(ras_in):
    # takes in path for georaster file, returns raster objects with following attributes:
        # ras.data -- numpy array of raster data
        # ras.gt -- geotransformation
        # ras.proj -- projection
        # ras.cols -- number of columns in raster
        # ras.rows -- number of rows in raster
        # ras.band -- raster band (1 only supported)
        # ras.T0 -- affine transformation for cell corners
        # ras.T1 -- affine transformation for cell centers

    # dependencies
    import gdal
    from affine import Affine
    import numpy as np

    #define class rasterObj
    class rasterObj(object):
        def __init__(self, raster):
            # load metadata
            self.gt = raster.GetGeoTransform()
            self.proj = raster.GetProjection()
            self.cols = raster.RasterXSize
            self.rows = raster.RasterYSize
            self.band_count = raster.RasterCount
            if self.band_count == 1:
                self.band = raster.GetRasterBand(1)
                self.data = self.band.ReadAsArray()
                self.no_data = self.band.GetNoDataValue()
            elif self.band_count > 1:
                self.band = []
                self.data = []
                for ii in range(1, self.band_count + 1):
                    self.band.append(raster.GetRasterBand(ii))
                    self.data.append(self.band[ii - 1].ReadAsArray())
                self.no_data = self.band[0].GetNoDataValue()
            # get affine transformation
            self.T0 = Affine.from_gdal(*raster.GetGeoTransform())
            # cell-centered affine transformation
            self.T1 = self.T0 * Affine.translation(0.5, 0.5)

    # open single band geo-raster file
    ras = gdal.Open(ras_in, gdal.GA_ReadOnly)

    # read data
    ras_out = rasterObj(ras)

    # close file
    ras = None

    return ras_out


# saves raster to file
def raster_save(ras_object, file_path, file_format="GTiff", data_format="float32"):
    # saves "ras_object" to "file_path" in "file_format"
    # file_format can be: "GTiff",
    # data_format can be: "float32", "float64", "byte", "int16", "int32", "uint16", "uint32"

    # dependencies
    import gdal
    import numpy as np

    if data_format == "float32":
        gdal_data_format = gdal.GDT_Float32
    elif data_format == "float64":
        gdal_data_format = gdal.GDT_Float64
    elif data_format == "byte":
        gdal_data_format = gdal.GDT_Byte
    elif data_format == "int16":
        gdal_data_format = gdal.GDT_Int16
    elif data_format == "int32":
        gdal_data_format = gdal.GDT_Int32
    elif data_format == "uint16":
        gdal_data_format = gdal.GDT_UInt16
    elif data_format == "uint32":
        gdal_data_format = gdal.GDT_UInt32
    else:
        raise Exception(data_format, 'is not a valid data_format.')

    # confirm band count matches data length
    if isinstance(ras_object.data, list):
        if ras_object.data.__len__() != ras_object.band_count:
            raise Exception("ras_object.band_count and length of ras_object.data do not agree.")
    elif isinstance(ras_object.data, np.ndarray):
        if ras_object.band_count == 1:
            if ras_object.shape.__len__() == 2:
                # nest data in list for single band output
                ras_object.data = [ras_object.data]
            else:
                raise Exception("2D array expected for ras_object.data and ras_object.band_count == 1")
        else:
            raise Exception("multi-band output as 3D array not yet supported. Consider passing as list of 2D arrays.")

    outdriver = gdal.GetDriverByName(file_format)
    outdata = outdriver.Create(file_path, ras_object.cols, ras_object.rows, ras_object.band_count, gdal_data_format)
    # Set metadata
    outdata.SetGeoTransform(ras_object.gt)
    outdata.SetProjection(ras_object.proj)

    # Write data for each band
    for ii in range(0, ras_object.band_count):
        outdata.GetRasterBand(ii + 1).WriteArray(ras_object.data[ii])
        outdata.GetRasterBand(ii + 1).SetNoDataValue(ras_object.no_data)

def ras_dif(ras_1_in, ras_2_in, inherit_from=1):
    # Returns raster object as follows:
        # ras_dif.data = ras_1. data - ras_2.data
        # metadata inherited from "inherit_from" (1 or 2).
    # Rasters must be identical in location, scale, and size

    # Dependencies
    import numpy as np

    if inherit_from == 1:
        ras_A = ras = raster_load(ras_1_in)
        ras_B = ras = raster_load(ras_2_in)
        flip_factor = 1
    elif inherit_from == 2:
        ras_A = ras = raster_load(ras_2_in)
        ras_B = ras = raster_load(ras_1_in)
        flip_factor = -1
    else:
        raise Exception('"inherit_from" must be either "1" or "2."')

    # check if identical origins and scales
    aff_dif = np.array(ras_A.T1) - np.array(ras_B.T1)

    if np.sum(np.abs(aff_dif)) != 0:
        raise Exception('Rasters are of different scales or origins, no difference was taken. Cell shifting may be needed!')

    ras_dif = ras_A
    # handle nas!
    mask = (ras_A.data == ras_A.no_data) | (ras_B.data == ras_B.no_data)
    ras_dif.data = (ras_A.data - ras_B.data) * flip_factor
    ras_dif.data[mask] = ras_A.no_data

    return ras_dif


def raster_burn(ras_in, shp_in, burn_val):
    # burns "burn_val" into "ras_in" where overlaps with "shp_in"

    # Dependencies
    import subprocess

    # convert burn_val to string
    burn_val = str(burn_val)

    # make gdal_rasterize command - will burn value to raster where polygon intersects
    cmd = 'gdal_rasterize -burn ' + burn_val + ' ' + shp_in + ' ' + ras_in

    # run command
    subprocess.call(cmd, shell=True)

def raster_merge(ras_in_dir, ras_in_ext, ras_out, no_data="-9999"):
    # merges all raster files in directory "ras_in_dir" with extention "ras_in_ext" and saves them as a merged output "ras_out"

    # Dependencies
    import subprocess
    from os import listdir, chdir

    dir_list = listdir(ras_in_dir)
    file_list = [k for k in dir_list if k.endswith(ras_in_ext)]
    file_str = ' '.join(file_list)

    # make gdal_rasterize command - will burn value to raster where polygon intersects
    cd_cmd = "Set-Location -Path " + ras_in_dir
    cmd = 'gdal_merge.py -init ' + no_data + ' -n ' + no_data + ' -a_nodata ' + no_data + ' -o ' + ras_out + ' ' + file_str

    chdir(ras_in_dir)
    # run command
    subprocess.call(cmd, shell=True)

def point_sample_raster(ras_in, pts_in, pts_out, pts_xcoord_name, pts_ycoord_name, sample_col_name, sample_no_data_value=-9999):

    # takes in csv file of points "pts_in" with x-column "pts_xcoord_name" and y-column "pts_ycoord_name" and saves
    # point values of raster "ras_in" to column "sample_col_name" in output csv "pts_out"

    # Dependencies
    import pandas as pd
    import numpy as np

    # read points
    pts = pd.read_csv(pts_in)
    # read raster
    ras = raster_load(ras_in)

    # convert point coords to raster index
    row_col_pts = np.rint(~ras.T1 * (pts[pts_xcoord_name], pts[pts_ycoord_name])).astype(int)
    row_col_pts = (row_col_pts[1], row_col_pts[0])

    # read raster values of points
    samples = ras.data[row_col_pts]

    # replace no_data values
    samples[samples == ras.no_data] = np.nan

    # add to pts df
    pts.loc[:, sample_col_name] = samples

    # write to file
    pts.to_csv(pts_out, index=False, na_rep=sample_no_data_value)

def raster_to_hdf5(ras_in, hdf5_out, data_col_name="data"):
    import numpy as np
    import vaex

    ras = raster_load(ras_in)

    row_map = np.full_like(ras.data, 0).astype(int)
    for ii in range(0, ras.rows):
        row_map[ii, :] = ii
    col_map = np.full_like(ras.data, 0).astype(int)
    for ii in range(0, ras.cols):
        col_map[:, ii] = ii

    index_x = np.reshape(row_map, [ras.rows * ras.cols])
    index_y = np.reshape(col_map, [ras.rows * ras.cols])
    coords = ras.T1 * (index_x, index_y)

    # add to vaex_df
    df = vaex.from_arrays(index_x=index_x, index_y=index_y, UTM11N_x=coords[0], UTM11N_y=coords[1])
    df.add_column(data_col_name, np.reshape(ras.data, [ras.rows * ras.cols]), dtype=None)

    # does not seem to work...
    # df.add_variable("no_data", ras.no_data, overwrite=True, unique=True)
    # df.set_variable("no_data", ras.no_data)

    # export to file
    df.export_hdf5(hdf5_out)

# something fishy here... it seems that the values are coming out in flipped coordinates
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
        ras = raster_load(ras_in[ii])

        # convert sample points to index refference
        row_col_pts = np.floor(~ras.T0 * (df.UTM11N_x.values, df.UTM11N_y.values)).astype(int)

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

# could clean up for time using KDTrees if desired. Currently takes +/- 100s for .25 res
def raster_nearest_neighbor(points, ras):
    import numpy as np

    # calculate min distance to tree and nearest tree index
    index_map = np.full_like(ras.data, ras.no_data)
    distance_map = np.full_like(ras.data, ras.no_data)
    # slow but works (60s?)
    for ii in range(0, ras.cols):
        for jj in range(0, ras.rows):
            cell_coords = ras.T1 * [ii, jj]
            distances = np.sqrt((cell_coords[0] - np.array(points.UTM11N_x))**2 + (cell_coords[1] - np.array(points.UTM11N_y))**2)
            nearest_id = np.argmin(distances)
            index_map[jj, ii] = points.index[nearest_id]
            distance_map[jj, ii] = distances[nearest_id]
    return index_map, distance_map



# ras.GetRasterBand(1)
# data = np.array(ras.ReadAsArray())

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# fig = plt.imshow(data[3, :, :], interpolation='nearest')
# plt.show(fig)