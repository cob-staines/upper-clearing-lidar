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

    # export to file
    df.export_hdf5(hdf5_out)


# something fishy here... it seems that the values are coming out in flipped coordinates
# def hdf5_sample_raster(hdf5_in, hdf5_out, ras_in, sample_col_name="sample"):
#     # can be single ras_in/sample_col_name or list of both
#     import numpy as np
#     import vaex
#
#     if (type(ras_in) == str) & (type(sample_col_name) == str):
#         # convert to list of length 1
#         ras_in = [ras_in]
#         sample_col_name = [sample_col_name]
#     elif (type(ras_in) == list) & (type(sample_col_name) == list):
#         if len(ras_in) != len(sample_col_name):
#             raise Exception('Lists of "ras_in" and "sample_col_name" are not the same length.')
#     else:
#         raise Exception('"ras_in" and "sample_col_name" are not consistent in length or format.')
#
#     # load hdf5_in
#     #df = vaex.open(hdf5_in, 'r+')
#     df = vaex.open(hdf5_in)
#
#     for ii in range(0, len(ras_in)):
#         # load raster
#         ras = raster_load(ras_in[ii])
#
#         # convert sample points to index reference
#         row_col_pts = np.floor(~ras.T0 * (df.UTM11N_x.values, df.UTM11N_y.values)).astype(int)
#
#         # flag samples out of raster bounds
#         outbound_x = (row_col_pts[0] < 0) | (row_col_pts[0] > (ras.rows - 1))
#         outbound_y = (row_col_pts[1] < 0) | (row_col_pts[1] > (ras.cols - 1))
#         outbound = outbound_x | outbound_y
#
#         # list of points in bounds
#         sample_pts = (row_col_pts[0][~outbound], row_col_pts[1][~outbound])
#
#         # read raster values of sample_points
#         samples = np.full(outbound.shape, ras.no_data)
#         samples[~outbound] = ras.data[sample_pts]
#
#         # add column to df
#         df.add_column(sample_col_name[ii], samples, dtype=None)
#
#         ras = None
#
#     # save to hdf5_out
#     df.export_hdf5(hdf5_out)
#     df.close()


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

    pts_coords = ras.T1 * pts_index
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

# def pd_sample_raster(parent, ras_parent, ras_child, colnames, include_nans=False):
#     import numpy as np
#
#     # sample child in child coords
#     child = raster_to_pd(ras_child, colnames, include_nans)
#
#     if parent is None:
#         # default to raster_to_pd output
#         return child
#     else:
#         # import if ras is file path, move on if ras is raster object
#         if not isinstance(ras_child, rasterObj):
#             if isinstance(ras_child, str):
#                 ras_child_in = ras_child
#                 ras_child = raster_load(ras_child_in)
#             else:
#                 raise Exception('ras is not an instance of rasterObj or str (filepath), raster_to_pd() aborted.')
#
#         if not isinstance(ras_parent, rasterObj):
#             if isinstance(ras_parent, str):
#                 ras_parent_in = ras_parent
#                 ras_parent = raster_load(ras_parent_in)
#             else:
#                 raise Exception('ras is not an instance of rasterObj or str (filepath), raster_to_pd() aborted.')
#
#         # confirmation transforms
#         # [y_index, x_index] = ~T * [x_coord, y_coord]
#         train = ~ras_parent.T1 * (parent.x_coord, parent.y_coord)
#         np.max(np.array(parent.x_index) - np.array(train[1]))
#         np.max(np.array(parent.y_index) - np.array(train[0]))
#
#         # [x_coord, y_coord] = T * [y_index, x_index]
#         peace = ras_parent.T1 * (parent.y_index, parent.x_index)
#         np.all(np.array(parent.x_coord) == np.array(peace[0]))
#         np.all(np.array(parent.y_coord) == np.array(peace[1]))
#
#         # [y_index, x_index] = ~T * [x_coord, y_coord]
#         train = ~ras_child.T1 * (child.x_coord, child.y_coord)
#         np.max(np.array(child.x_index) - np.array(train[1]))
#         np.max(np.array(child.y_index) - np.array(train[0]))
#
#         # [x_coord, y_coord] = T * [y_index, x_index]
#         peace = ras_child.T1 * (child.y_index, child.x_index)
#         np.all(np.array(child.x_coord) == np.array(peace[0]))
#         np.all(np.array(child.y_coord) == np.array(peace[1]))
#
#         # parent coord into child and back
#         # [y_index, x_index] = ~T * [x_coord, y_coord]
#         train = ~ras_child.T0 * (parent.x_coord, parent.y_coord)
#         peace = ras_child.T0 * train
#         np.max(np.array(parent.x_coord) - np.array(peace[0]))
#         np.max(np.array(parent.y_coord) - np.array(peace[1]))
#
#         # parent coords into child and back
#         # [x_coord, y_coord] = T * [y_index, x_index]
#         peace = ras_child.T0 * (parent.y_index, parent.x_index)
#         train = ~ras_child.T0 * peace
#         np.max(np.array(parent.x_index) - np.array(train[1]))
#         np.max(np.array(parent.y_index) - np.array(train[0]))
#
#
#
#         # convert parent coords to child index
#         parent_in_child_index = ~ras_child.T0 * (parent.x_coord, parent.y_coord)
#         parent.loc[:, 'child_x_index'] = np.floor(parent_in_child_index[1]).astype(int)
#         parent.loc[:, 'child_y_index'] = np.floor(parent_in_child_index[0]).astype(int)
#
#         parent.loc[parent.child_x_index < 0, 'child_x_index'] = np.nan
#         parent.loc[parent.child_x_index >= ras_child.cols, 'child_x_index'] = np.nan
#         parent.loc[parent.child_y_index < 0, 'child_x_index'] = np.nan
#         parent.loc[parent.child_y_index >= ras_child.rows, 'child_x_index'] = np.nan
#
#
#         # # convert child coords to parent index
#         # child_in_parent_index = ~ras_parent.T0 * (child.x_coord, child.y_coord)
#         # child.loc[:, 'parent_x_index'] = np.floor(child_in_parent_index[1]).astype(int)
#         # child.loc[:, 'parent_y_index'] = np.floor(child_in_parent_index[0]).astype(int)
#
#         # instead of merge, can we try just logical indexing?
#         valid = ~np.isnan(parent.child_y_index) & ~np.isnan(parent.child_x_index)
#         parent.loc[:, 'child_values'] = np.nan
#         parent.loc[valid, 'child_values'] = ras_child.data[parent.child_x_index.loc[valid].astype(int), parent.child_y_index.loc[valid].astype(int)]
#
#         # drop unnecessary columns before merge
#         # child = child.drop(columns=['x_coord', 'y_coord'])
#         # merge along child index
#         # both give same results, but values still disagree
#         pc = parent.merge(child, how='left', left_on=['child_x_index', 'child_y_index'], right_on=['x_index', 'y_index'], suffixes=['', '_child'])
#         # pc2 = parent.merge(child, how='left', left_on=['x_index', 'y_index'], right_on=['parent_x_index', 'parent_y_index'], suffixes=['', '_child'])  # neither of these methods produce results that agree. Issues need to be resolved.
#
#         # drop child index
#         # pc = pc.drop(columns=['child_x_index', 'child_y_index', 'x_index_child', 'y_index_child'])
#
#         return pc


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
        valid_cells = np.where(n_count > n_threshold)
        invalid_cells = np.where(n_count <= n_threshold)
    else:
        valid_cells = np.where(np.isnan(values))
        invalid_cells = np.where(~np.isnan(values))

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