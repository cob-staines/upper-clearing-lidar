# import rasterio
# import ogr
import numpy as np

#define class rasterObj
class rasterObj(object):
    def __init__(self, raster):
        from affine import Affine

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

    import numpy as np

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
            if ras_object.data.shape.__len__() == 2:
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

    del outdata  # Flush



def raster_dif(ras_1_in, ras_2_in, inherit_from=1, dif_out=None):
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

    if dif_out is not None:
        # output dif
        raster_save(ras_dif, dif_out, data_format='float32')

    return ras_dif


def raster_dif_gdal(ras_1_in, ras_2_in, dif_out=None, inherit_from=1):
    # Returns raster object/saves file as follows:
    # ras_dif.data = ras_1.data - ras_2.data
    # metadata inherited from "inherit_from" (1 or 2).

    if inherit_from == 1:
        flip_factor = 1
        ras_a_in = ras_1_in
        ras_b_in = ras_2_in
    elif inherit_from == 2:
        flip_factor = -1
        ras_a_in = ras_2_in
        ras_b_in = ras_1_in
    else:
        raise Exception('inherit_from must take values 1 or 2 only.')

    # load ras_a
    ras_dif = raster_load(ras_a_in)

    # load ras_b reprojected to ras_a
    data_a = ras_dif.data.copy()
    data_b = gdal_raster_reproject(ras_b_in, ras_a_in)

    # take difference
    ras_dif.data = (data_a - data_b) * flip_factor

    if dif_out is not None:
        # export to file
        raster_save(ras_dif, dif_out)

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


ras_in_dir = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_045\\19_045_las_proc\\TEMP_FILES\\12_dem\\res_.10'
ras_in_ext = '.bil'
ras_out = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_045\\19_045_las_proc\\OUTPUT_FILES\\DEM\\19_045_dem_r.10.tif'

def raster_merge(ras_in_dir, ras_in_ext, ras_out, no_data="-9999"):
    # merges all raster files in directory "ras_in_dir" with extention "ras_in_ext" and saves them as a merged output "ras_out"

    # Dependencies
    import subprocess
    from os import listdir, chdir

    dir_list = listdir(ras_in_dir)
    file_list = [k for k in dir_list if k.endswith(ras_in_ext)]
    file_str = ' '.join(file_list)

    # make gdal_rasterize command - will burn value to raster where polygon intersects
    #cd_cmd = "Set-Location -Path " + ras_in_dir
    cmd = 'gdal_merge.py -init ' + no_data + ' -n ' + no_data + ' -a_nodata ' + no_data + ' -o ' + ras_out + ' ' + file_str

    chdir(ras_in_dir)
    # run command
    subprocess.call(cmd, shell=True)


def csv_sample_raster(ras_in, pts_in, pts_out, pts_xcoord_name, pts_ycoord_name, sample_col_name, sample_no_data_value=''):

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

# rewrite if needed using raster to pd, opd to hdf5, no vaex dependence
# def raster_to_hdf5(ras_in, hdf5_out, data_col_name="data"):
#     import numpy as np
#     import vaex
#
#     ras = raster_load(ras_in)
#
#     row_map = np.full_like(ras.data, 0).astype(int)
#     for ii in range(0, ras.rows):
#         row_map[ii, :] = ii
#     col_map = np.full_like(ras.data, 0).astype(int)
#     for ii in range(0, ras.cols):
#         col_map[:, ii] = ii
#
#     index_x = np.reshape(col_map, [ras.rows * ras.cols])
#     index_y = np.reshape(row_map, [ras.rows * ras.cols])
#     coords = ras.T1 * (index_x, index_y)
#
#     # add to vaex_df
#     df = vaex.from_arrays(index_x=index_x, index_y=index_y, UTM11N_x=coords[0], UTM11N_y=coords[1])
#     df.add_column(data_col_name, np.reshape(ras.data, [ras.rows * ras.cols]), dtype=None)
#
#     # export to file
#     df.export_hdf5(hdf5_out)


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


def raster_to_pd(ras, colnames, include_nans=False):
    import numpy as np
    import pandas as pd

    # test if ras is path or raster_object
    if not isinstance(ras, rasterObj):
        if isinstance(ras, str):
            ras_in = ras
            ras = raster_load(ras_in)
        else:
            raise Exception('ras is not an instance of rasterObj or str (filepath), raster_to_pd() aborted.')

    if isinstance(ras.data, np.ndarray):
        # nest data in list if not already
        ras.data = [ras.data]
    if ras.band_count != len(ras.data):
        raise Exception('data dimensions do not match band_count, raster_to_pd() aborted.')

    if isinstance(colnames, str):
        # nest colname in list if not already
        colnames = [colnames]
    if ras.band_count != len(colnames):
        raise Exception('length of colname does not match band_count, raster_to_pd() aborted.')

    all_vals = np.full([ras.rows, ras.cols], True)

    nan_vals = np.full([ras.rows, ras.cols, ras.band_count], False)
    for ii in range(0, ras.band_count):
        nan_vals[:, :, ii] = (ras.data[ii] == ras.no_data)
    nan_vals = np.any(nan_vals, axis=2)

    if include_nans:
        pts_index = np.where(all_vals)
    else:
        # only non nans
        pts_index = np.where(~nan_vals)

    pts_flop = (pts_index[1], pts_index[0])
    pts_coords = ras.T1 * pts_flop  # flop from numpy (y, x) to affine (x, y)
    pts = pd.DataFrame({'x_coord': pts_coords[0],  # affine transform output returns [x, y]
                        'y_coord': pts_coords[1],
                        'x_index': pts_index[1],  # numpy output from np.where() returns [y, x]
                        'y_index': pts_index[0]})

    for ii in range(0, ras.band_count):
        pts.loc[:, colnames[ii]] = ras.data[ii][pts.y_index, pts.x_index]
        # sub in np.nan for no_data values
        pts.loc[pts.loc[:, colnames[ii]] == ras.no_data, colnames[ii]] = np.nan

    return pts


def gdal_raster_reproject(src, match, nodatavalue=np.nan):
    from osgeo import gdal, gdalconst
    import numpy as np

    # Source
    if isinstance(src, str):
        src_filename = src
        src = gdal.Open(src_filename, gdalconst.GA_ReadOnly)
    elif ~isinstance(src, gdal.Dataset):
        raise Exception('src is not either a file path or osgeo.gdal.Dataset')

    src_proj = src.GetProjection()
    src_geotrans = src.GetGeoTransform()
    band_count = src.RasterCount

    # Match
    if isinstance(match, str):
        match_filename = match
        match = gdal.Open(match_filename, gdalconst.GA_ReadOnly)
    elif ~isinstance(match, gdal.Dataset):
        raise Exception('match is not either a file path or osgeo.gdal.Dataset')

    # We want a section of source that matches this:
    match_proj = match.GetProjection()
    match_geotrans = match.GetGeoTransform()
    wide = match.RasterXSize
    high = match.RasterYSize

    # create memory destination
    mem_drv = gdal.GetDriverByName('MEM')
    dest = mem_drv.Create('', wide, high, band_count, gdal.GDT_Float32)
    # pad with nodatavalue
    for ii in range(1, band_count + 1):
        dest.GetRasterBand(ii).WriteArray(np.full((high, wide), nodatavalue), 0, 0)

    # Set the geotransform
    dest.SetGeoTransform(match_geotrans)
    dest.SetProjection(match_proj)
    # Perform the projection/resampling
    # res = gdal.ReprojectImage(src, dest, src_proj, match_proj, gdal.GRA_Bilinear)
    res = gdal.ReprojectImage(src, dest, src_proj, match_proj, gdal.GRA_NearestNeighbour)

    rp_array = np.full((high, wide, band_count), nodatavalue)
    for ii in range(1, band_count + 1):
        rp_array[:, :, ii - 1] = np.array(dest.GetRasterBand(ii).ReadAsArray())

    del dest  # Flush

    return rp_array


def pd_sample_raster_gdal(data_dict, include_nans=False, nodatavalue=np.nan):
    files = list(data_dict.values())
    colnames = list(data_dict.keys())

    # take first item in files as parent
    print('Loading ' + str(colnames[0]) + "... ", end='')
    df = raster_to_pd(files[0], colnames[0], include_nans=include_nans)
    print('done')

    # for remaining items in files
    for ii in range(1, len(files)):
        print('Loading ' + colnames[ii] + "... ", end='')
        rs_array = gdal_raster_reproject(files[ii], files[0], nodatavalue=nodatavalue)

        band_count = rs_array.shape[2]
        if band_count == 1:
            df.loc[:, colnames[ii]] = rs_array[df.y_index, df.x_index, 0]
        elif band_count > 1:
            if len(colnames[ii]) != band_count:
                raise Exception('colnames key ' + str(colnames[ii]) + ' does not agree with number of bands in image.')
            for jj in range(0, band_count):
                df.loc[:, colnames[ii][jj]] = rs_array[df.y_index, df.x_index, jj]

        print('done')
    return df


def delauney_fill(values, values_out, ras_template, n_count=None, n_threshold=0):
    import numpy as np
    from scipy.interpolate import LinearNDInterpolator

    if (n_threshold > 0) & (n_count is None):
        raise Exception('no n_count provided, could not threshold to min_n')

    if isinstance(values, str):
        values_in = values
        ras = raster_load(values_in)
        values = ras.data
        values[values == ras.no_data] = np.nan

    if isinstance(n_count, str):
        n_count_in = n_count
        ras = raster_load(n_count_in)
        n_count = ras.data

    # delauney triangulation between cells where n > min_n
    if n_threshold > 0:
        # ignoring nan values...
        valid_cells = np.where(n_count > n_threshold)
        invalid_cells = np.where(n_count <= n_threshold)
    else:
        valid = ~np.isnan(values)
        valid_cells = np.where(valid)
        invalid_cells = np.where(~valid)

    # unzip to coords
    valid_coords = list(zip(valid_cells[0], valid_cells[1]))
    # create interpolation function
    delauney_int = LinearNDInterpolator(valid_coords, values[valid_cells])

    values_filled = values.copy()
    # interpolate invalid cells
    values_filled[invalid_cells] = delauney_int(invalid_cells)

    # export filled to raster
    val_ras = raster_load(ras_template)
    val_ras.data = values_filled
    val_ras.data[np.isnan(val_ras.data)] = val_ras.no_data
    raster_save(val_ras, values_out, data_format='float32')

    return values_filled


# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# fig = plt.imshow(data[3, :, :], interpolation='nearest')
# plt.show(fig)