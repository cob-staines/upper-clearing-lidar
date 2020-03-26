import pandas as pd
import numpy as np
from raster_load import raster_load

# Config
ras_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_all_200311\\OUTPUT_FILES\\DEM\\19_149_all_200311_628000_5646525dem_.25m.bil"
pts_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\surveys\\all_ground_points_UTM11N.csv"
pts_out = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_all_200311\\OUTPUT_FILES\\DEM\\19_149_all_200311_628000_5646525dem_.25m_s2_ground_pount_samples.csv"

pts_xcoord_name = "xcoordUTM11"
pts_ycoord_name = "ycoordUTM11"
sample_col_name = "r.04s8"
sample_no_data_value = ""

def point_sample_raster(ras_in, pts_in, pts_out, pts_xcoord_name, pts_ycoord_name, sample_col_name, sample_no_data_value):
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

# manual automation
pts_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\surveys\\all_ground_points_UTM11N_uid_flagged_cover.csv"

step = 2
step_name = "2"
offset = (.01, .03, .04, .05, .1, .2)
offset_name = (".01", ".03", ".04", ".05", ".1", ".2")

pts_xcoord_name = "xcoordUTM11"
pts_ycoord_name = "ycoordUTM11"
sample_no_data_value = ""

pts_out = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_all_200311\\OUTPUT_FILES\\DEM\\offset_opt\\19_149_all_200311_628000_5646525dem_.04_step_" + step_name + "_ground_pount_samples.csv"

pts = pd.read_csv(pts_in)

for ii in range(0, 6):
    ras_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_all_200311\\OUTPUT_FILES\\DEM\\offset_opt\\19_149_all_200311_628000_5646525dem_.04_step_" + step_name + "_offset_" + offset_name[ii] + ".bil"
    sample_col_name = "s" + step_name + "_o" + offset_name[ii]

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
