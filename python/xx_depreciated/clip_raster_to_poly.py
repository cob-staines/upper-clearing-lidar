raster = rasin

# Read points from shapefile
shpin = """C:\\Users\\Cob\\index\\educational\\usask\\research\\dronefest\\data\\19_133_sd_analysis\\DroneFest_surface_types_WGS84UTMzone13N.shp"""
rasin = """C:\\Users\\Cob\\index\\educational\\usask\\research\\dronefest\\data\\19_133_sd_analysis\\sfm_minu_lidar.tif"""

pts = gpd.read_file(shpin)
pts = pts[['id', 'surf_type', 'geometry']]
pts.index = range(len(pts))

# converts coordinates to index
def bbox2ix(bbox,gt):
    xo = int(round((bbox[0] - gt[0])/gt[1]))
    yo = int(round((gt[3] - bbox[3])/gt[1]))
    xd = int(round((bbox[1] - bbox[0])/gt[1]))
    yd = int(round((bbox[3] - bbox[2])/gt[1]))
    return(xo,yo,xd,yd)
ras = rasin
shp = shpin
def rasclip(ras,shp):
    ds = gdal.Open(ras)
    gt = ds.GetGeoTransform()

    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(shp, 0)
    layer = dataSource.GetLayer()

    for feature in layer:

        xo,yo,xd,yd = bbox2ix(feature.GetGeometryRef().GetEnvelope(),gt)
        arr = ds.ReadAsArray(xo,yo,xd,yd)
        yield arr

    layer.ResetReading()
    ds = None
    dataSource = None

gen = rasclip(rasin,shpin)


def get_aoi_intersection(raster, aoi):
    """
    Returns a wkbPolygon geometry with the intersection of a raster and a shpefile containing an area of interest

    Parameters
    ----------
    raster
        A raster containing image data
    aoi
        A shapefile with a single layer and feature
    Returns
    -------
    a ogr.Geometry object containing a single polygon with the area of intersection

    """
    raster_shape = get_raster_bounds(raster)
    aoi.GetLayer(0).ResetReading()  # Just in case the aoi has been accessed by something else
    aoi_feature = aoi.GetLayer(0).GetFeature(0)
    aoi_geometry = aoi_feature.GetGeometryRef()
    return aoi_geometry.Intersection(raster_shape)

peace = get_aoi_intersection(rasin, shpin)

# open raster file
ras = gdal.Open(rasin)
band = ras.GetRasterBand(1)

gt = ras.GetGeoTransform()
proj = ras.GetProjection()
arr = ras.ReadAsArray()

driver = ogr.GetDriverByName('ESRI Shapefile')
data_source = driver.Open(shpin, 0)
if data_source is None:
    report_and_exit("File read failed: %s", vector_data_path)
layer = data_source.GetLayer(0)

driver = gdal.GetDriverByName('MEM')
cols = ras.RasterXSize
rows = ras.RasterYSize
output_fname = """C:\\Users\\Cob\\index\\educational\\usask\\research\\dronefest\\data\\19_133_sd_analysis\\DroneFest_surface_types_mask_WGS84UTMzone13N.tif"""

target_ds = driver.Create(output_fname, cols, rows, 1, gdal.GDT_Byte)
target_ds.SetGeoTransform(gt)
target_ds.SetProjection(proj)
gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[1])
target_ds = None

peace = target_ds.ReadAsArray()


# outdata.GetRasterBand(1).WriteArray(somedatahere)