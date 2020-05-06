# ras_dif takes in raster file A and B and returns the difference (A-B)
from raster_load import raster_load
from raster_save import raster_save
import numpy as np

def ras_dif(raster_1, raster_2, inherit_from=1):

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

ras_1_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_045\\19_045_snow_on\\OUTPUT_FILES\\DEM\\offset_opt\\19_045_all_200311_628000_5646525dem_.10_step_1_offset_.04.bil"
ras_2_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\DEM\\19_149_all_200311_628000_5646525dem_.10m.bil"

dif_out = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_045\\19_045_snow_on\\OUTPUT_FILES\\DEM\\offset_opt\\19_045_all_200311_628000_5646525dif_.10_step_1_offset_.04.tif"

peace = ras_dif(ras_1_in, ras_2_in, inherit_from=2)
raster_save(peace, dif_out, "GTiff")
