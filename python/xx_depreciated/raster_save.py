# saves raster to file
def raster_save(ras_object, file_path, file_type):
    # filetype can be: "GTiff",
    # dependencies
    import gdal

    outdriver = gdal.GetDriverByName(file_type)
    outdata = outdriver.Create(file_path, ras_object.cols, ras_object.rows, 1, gdal.GDT_Float32)
    # Set metadata
    outdata.SetGeoTransform(ras_object.gt)
    outdata.SetProjection(ras_object.proj)

    # Write data
    outdata.GetRasterBand(1).WriteArray(ras_object.data)
    outdata.GetRasterBand(1).SetNoDataValue(ras_object.no_data)
    outdata = None